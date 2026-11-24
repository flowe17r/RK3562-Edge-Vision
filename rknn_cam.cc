// V4L2 YUYV -> RGB letterbox 640x640 -> RKNN -> postprocess -> optional MJPEG over HTTP
#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <linux/fb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "postprocess.h"
#include "rknn_api.h"
#include "turbojpeg.h"

#define BUF_NUM 4
#define PAD_GRAY 114

struct cam {
    int fd;
    unsigned int nbuf;
    void *start[BUF_NUM];
    size_t length[BUF_NUM];
    int w, h;
};

struct stat_acc {
    double sum = 0.0, mn = 0.0, mx = 0.0;
    int n = 0;
    std::vector<double> smp;
    void add(double v) {
        if (n == 0 || v < mn) mn = v;
        if (n == 0 || v > mx) mx = v;
        sum += v;
        n++;
        if (smp.size() < 4000) smp.push_back(v);
    }
    double avg() const { return n ? sum / n : 0.0; }
    double pct(double p) const {
        if (smp.empty()) return 0.0;
        std::vector<double> t = smp;
        std::sort(t.begin(), t.end());
        size_t i = (size_t)(p * (t.size() - 1) + 0.5);
        return t[i < t.size() ? i : t.size() - 1];
    }
};

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

static int xioctl(int fd, unsigned long req, void *arg) {
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

static int cam_open(struct cam *c, const char *dev, int w, int h, int fps) {
    memset(c, 0, sizeof(*c));
    c->fd = open(dev, O_RDWR | O_NONBLOCK);
    if (c->fd < 0) {
        printf("open %s failed: %s\n", dev, strerror(errno));
        return -1;
    }
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = w;
    fmt.fmt.pix.height = h;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(c->fd, VIDIOC_S_FMT, &fmt) < 0) {
        printf("VIDIOC_S_FMT failed: %s\n", strerror(errno));
        return -1;
    }
    c->w = fmt.fmt.pix.width;
    c->h = fmt.fmt.pix.height;

    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps;
    xioctl(c->fd, VIDIOC_S_PARM, &parm);

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = BUF_NUM;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(c->fd, VIDIOC_REQBUFS, &req) < 0) {
        printf("VIDIOC_REQBUFS failed: %s\n", strerror(errno));
        return -1;
    }
    c->nbuf = req.count;
    for (unsigned int i = 0; i < c->nbuf; i++) {
        struct v4l2_buffer b;
        memset(&b, 0, sizeof(b));
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index = i;
        if (xioctl(c->fd, VIDIOC_QUERYBUF, &b) < 0) {
            printf("VIDIOC_QUERYBUF failed: %s\n", strerror(errno));
            return -1;
        }
        c->length[i] = b.length;
        c->start[i] = mmap(NULL, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, c->fd, b.m.offset);
        if (c->start[i] == MAP_FAILED) {
            printf("mmap failed: %s\n", strerror(errno));
            return -1;
        }
        xioctl(c->fd, VIDIOC_QBUF, &b);
    }
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(c->fd, VIDIOC_STREAMON, &type) < 0) {
        printf("VIDIOC_STREAMON failed: %s\n", strerror(errno));
        return -1;
    }
    printf("camera %s %dx%d YUYV, %u buffers\n", dev, c->w, c->h, c->nbuf);
    return 0;
}

static void cam_close(struct cam *c) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(c->fd, VIDIOC_STREAMOFF, &type);
    for (unsigned int i = 0; i < c->nbuf; i++) {
        if (c->start[i] && c->start[i] != MAP_FAILED) munmap(c->start[i], c->length[i]);
    }
    close(c->fd);
}

static int cam_dq(struct cam *c, size_t *bytes) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(c->fd, &fds);
    struct timeval tv;
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    int r = select(c->fd + 1, &fds, NULL, NULL, &tv);
    if (r <= 0) {
        if (r == 0) printf("select timeout\n");
        return -1;
    }
    struct v4l2_buffer b;
    memset(&b, 0, sizeof(b));
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    if (xioctl(c->fd, VIDIOC_DQBUF, &b) < 0) {
        if (errno != EAGAIN) printf("VIDIOC_DQBUF failed: %s\n", strerror(errno));
        return -1;
    }
    *bytes = b.bytesused;
    return (int)b.index;
}

static void cam_q(struct cam *c, int index);

// Always hand back the newest finished frame: drain anything already queued so latency cannot build up.
static int cam_dq_latest(struct cam *c, size_t *bytes, int *dropped) {
    int idx = cam_dq(c, bytes);
    if (idx < 0) return -1;
    for (;;) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(c->fd, &fds);
        struct timeval tv = {0, 0};
        if (select(c->fd + 1, &fds, NULL, NULL, &tv) <= 0) break;
        size_t nb = 0;
        int nidx = cam_dq(c, &nb);
        if (nidx < 0) break;
        cam_q(c, idx);
        idx = nidx;
        *bytes = nb;
        (*dropped)++;
    }
    return idx;
}

static void cam_q(struct cam *c, int index) {
    struct v4l2_buffer b;
    memset(&b, 0, sizeof(b));
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = index;
    xioctl(c->fd, VIDIOC_QBUF, &b);
}

static inline int clamp255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

static void yuyv_to_rgb_letterbox(const unsigned char *src, int sw, int sh, unsigned char *dst, int dst_w, int dst_h) {
    memset(dst, PAD_GRAY, (size_t)dst_w * dst_h * 3);
    int off_x = (dst_w - sw) / 2;
    int off_y = (dst_h - sh) / 2;
    if (off_x < 0) off_x = 0;
    if (off_y < 0) off_y = 0;

    for (int y = 0; y < sh && (y + off_y) < dst_h; y++) {
        const unsigned char *row = src + (size_t)y * sw * 2;
        unsigned char *out = dst + ((size_t)(y + off_y) * dst_w + off_x) * 3;
        for (int x = 0; x < sw; x += 2) {
            int y0 = row[0], u = row[1] - 128, y1 = row[2], v = row[3] - 128;
            row += 4;
            int r0 = (298 * y0 + 409 * v + 128) >> 8;
            int g0 = (298 * y0 - 100 * u - 208 * v + 128) >> 8;
            int b0 = (298 * y0 + 516 * u + 128) >> 8;
            int r1 = (298 * y1 + 409 * v + 128) >> 8;
            int g1 = (298 * y1 - 100 * u - 208 * v + 128) >> 8;
            int b1 = (298 * y1 + 516 * u + 128) >> 8;
            if (x + off_x < dst_w) {
                out[0] = clamp255(r0);
                out[1] = clamp255(g0);
                out[2] = clamp255(b0);
                if (x + 1 + off_x < dst_w) {
                    out[3] = clamp255(r1);
                    out[4] = clamp255(g1);
                    out[5] = clamp255(b1);
                }
            }
            out += 6;
        }
    }
}

static const unsigned char PALETTE[][3] = {
    {255, 64, 64},  {64, 255, 64},  {64, 64, 255},  {255, 255, 64},  {255, 64, 255},
    {64, 255, 255}, {255, 160, 64}, {160, 64, 255}, {64, 160, 255}, {255, 255, 255},
};

static void draw_rect(unsigned char *rgb, int w, int h, int x0, int y0, int x1, int y1, const unsigned char *color,
                      int thick) {
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 >= w) x1 = w - 1;
    if (y1 >= h) y1 = h - 1;
    if (x1 <= x0 || y1 <= y0) return;
    for (int t = 0; t < thick; t++) {
        int yy0 = y0 + t, yy1 = y1 - t, xx0 = x0 + t, xx1 = x1 - t;
        if (yy0 > yy1 || xx0 > xx1) break;
        for (int x = xx0; x <= xx1; x++) {
            unsigned char *p = rgb + ((size_t)yy0 * w + x) * 3;
            p[0] = color[0];
            p[1] = color[1];
            p[2] = color[2];
            p = rgb + ((size_t)yy1 * w + x) * 3;
            p[0] = color[0];
            p[1] = color[1];
            p[2] = color[2];
        }
        for (int y = yy0; y <= yy1; y++) {
            unsigned char *p = rgb + ((size_t)y * w + xx0) * 3;
            p[0] = color[0];
            p[1] = color[1];
            p[2] = color[2];
            p = rgb + ((size_t)y * w + xx1) * 3;
            p[0] = color[0];
            p[1] = color[1];
            p[2] = color[2];
        }
    }
}

struct glyph5x8 {
    char c;
    unsigned char rows[8];
};

// 5x8 bitmap font, bit4 = leftmost pixel (Buildroot 无 freetype，自带点阵)
static const glyph5x8 FONT5X8[] = {
    {' ', {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
    {'%', {0x19, 0x1A, 0x02, 0x04, 0x08, 0x0B, 0x13, 0x00}},
    {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00}},
    {':', {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00, 0x00}},
    {'-', {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x00}},
    {'/', {0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10, 0x00}},
    {'(', {0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02, 0x00}},
    {')', {0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08, 0x00}},
    {',', {0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x04, 0x08}},
    {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E, 0x00}},
    {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00}},
    {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F, 0x00}},
    {'3', {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E, 0x00}},
    {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02, 0x00}},
    {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E, 0x00}},
    {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E, 0x00}},
    {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08, 0x00}},
    {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E, 0x00}},
    {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C, 0x00}},
    {'a', {0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F, 0x00}},
    {'b', {0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E, 0x00}},
    {'c', {0x00, 0x00, 0x0E, 0x11, 0x10, 0x11, 0x0E, 0x00}},
    {'d', {0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F, 0x00}},
    {'e', {0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E, 0x00}},
    {'f', {0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08, 0x00}},
    {'g', {0x00, 0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x0E}},
    {'h', {0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x11, 0x00}},
    {'i', {0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E, 0x00}},
    {'j', {0x02, 0x00, 0x06, 0x02, 0x02, 0x02, 0x12, 0x0C}},
    {'k', {0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12, 0x00}},
    {'l', {0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00}},
    {'m', {0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11, 0x00}},
    {'n', {0x00, 0x00, 0x1E, 0x11, 0x11, 0x11, 0x11, 0x00}},
    {'o', {0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E, 0x00}},
    {'p', {0x00, 0x00, 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10}},
    {'q', {0x00, 0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x01}},
    {'r', {0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10, 0x00}},
    {'s', {0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E, 0x00}},
    {'t', {0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06, 0x00}},
    {'u', {0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D, 0x00}},
    {'v', {0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04, 0x00}},
    {'w', {0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A, 0x00}},
    {'x', {0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x00}},
    {'y', {0x00, 0x00, 0x11, 0x11, 0x11, 0x0F, 0x01, 0x0E}},
    {'z', {0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F, 0x00}},
};

static const unsigned char *glyph_rows(char c) {
    for (size_t i = 0; i < sizeof(FONT5X8) / sizeof(FONT5X8[0]); i++) {
        if (FONT5X8[i].c == c) return FONT5X8[i].rows;
    }
    return FONT5X8[0].rows;
}

// 5x8 点阵，scale=2 时字高 16px；黑底 + fg 色文字，越界自动裁剪
static void draw_text(unsigned char *rgb, int w, int h, int x, int y, const char *s, const unsigned char *fg,
                      int scale) {
    if (!s || !*s || scale < 1) return;
    int n = (int)strlen(s);
    int tw = n * 6 * scale - scale;
    int th = 8 * scale;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + tw > w) tw = w - x;
    if (y + th > h) th = h - y;
    if (tw <= 0 || th <= 0) return;
    for (int yy = 0; yy < th; yy++) {
        unsigned char *p = rgb + ((size_t)(y + yy) * w + x) * 3;
        memset(p, 0, (size_t)tw * 3);
    }
    for (int i = 0; i < n; i++) {
        const unsigned char *g = glyph_rows(s[i]);
        int gx = x + i * 6 * scale;
        if (gx >= x + tw) break;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 5; c++) {
                if (!(g[r] & (1 << (4 - c)))) continue;
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        int px = gx + c * scale + sx;
                        int py = y + r * scale + sy;
                        if (px >= x + tw || py >= y + th) continue;
                        unsigned char *p = rgb + ((size_t)py * w + px) * 3;
                        p[0] = fg[0];
                        p[1] = fg[1];
                        p[2] = fg[2];
                    }
                }
            }
        }
    }
}

struct person_track {
    int x0, y0, x1, y1;
    int hits, misses, age;
    bool active, counted, seen;
};

static float box_iou(int ax0, int ay0, int ax1, int ay1, int bx0, int by0, int bx1, int by1) {
    int ix0 = ax0 > bx0 ? ax0 : bx0;
    int iy0 = ay0 > by0 ? ay0 : by0;
    int ix1 = ax1 < bx1 ? ax1 : bx1;
    int iy1 = ay1 < by1 ? ay1 : by1;
    if (ix1 <= ix0 || iy1 <= iy0) return 0.0f;
    float in = (float)(ix1 - ix0) * (iy1 - iy0);
    float au = (float)(ax1 - ax0) * (ay1 - ay0) + (float)(bx1 - bx0) * (by1 - by0) - in;
    return au > 0 ? in / au : 0.0f;
}

static void log_person(int id) {
    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tmv);
    printf("%s person #%d\n", ts, id);
    FILE *f = fopen("count.log", "a");
    if (!f) return;
    fprintf(f, "%s person #%d\n", ts, id);
    fclose(f);
}

static int save_jpeg(const unsigned char *rgb, int w, int h, const char *path) {
    tjhandle tj = tjInitCompress();
    if (!tj) return -1;
    unsigned char *buf = nullptr;
    unsigned long sz = 0;
    int ret = -1;
    if (tjCompress2(tj, rgb, w, w * 3, h, TJPF_RGB, &buf, &sz, TJSAMP_420, 90, 0) == 0) {
        FILE *f = fopen(path, "wb");
        if (f) {
            if (fwrite(buf, 1, sz, f) == sz) ret = 0;
            fclose(f);
        }
    }
    tjFree(buf);
    tjDestroy(tj);
    return ret;
}

// Offline determinism check: run a folder of JPEGs through one model, one CSV row per frame.
static int eval_dir(rknn_context ctx, const char *model_path, const char *dir, int in_w, int in_h,
                    std::vector<int32_t> &zps, std::vector<float> &scales, uint32_t nout, int max_frames) {
    DIR *d = opendir(dir);
    if (!d) {
        printf("open dir %s failed: %s\n", dir, strerror(errno));
        return 1;
    }
    std::vector<std::string> names;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const char *n = de->d_name;
        size_t l = strlen(n);
        if (l > 4 && strcmp(n + l - 4, ".jpg") == 0) names.push_back(std::string(dir) + "/" + n);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    if (names.empty()) {
        printf("no .jpg in %s\n", dir);
        return 1;
    }

    const char *base = strrchr(model_path, '/');
    base = base ? base + 1 : model_path;
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "eval_%s.csv", base);
    char *dot = strrchr(csv_path, '.');
    if (dot) strcpy(dot, ".csv");
    FILE *csv = fopen(csv_path, "w");
    if (csv) fprintf(csv, "frame,det,person,best_prop,top\n");

    tjhandle tj = tjInitDecompress();
    std::vector<unsigned char> img((size_t)in_w * in_h * 3);
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.size = (uint32_t)img.size();
    in.fmt = RKNN_TENSOR_NHWC;
    in.pass_through = 0;
    std::vector<rknn_output> outs(nout);
    for (uint32_t i = 0; i < nout; i++) outs[i].want_float = 0;

    int ok = 0, ndet = 0, nperson = 0, nwith = 0, nprop = 0;
    double sum_prop = 0, sum_run = 0;
    for (size_t k = 0; k < names.size(); k++) {
        if (g_stop || (max_frames > 0 && (int)k >= max_frames)) break;
        FILE *f = fopen(names[k].c_str(), "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<unsigned char> jp(sz > 0 ? (size_t)sz : 1);
        size_t got = fread(jp.data(), 1, jp.size(), f);
        fclose(f);
        if (got != jp.size()) continue;

        int jw = 0, jh = 0, js = 0, jcs = 0;
        if (tjDecompressHeader3(tj, jp.data(), (unsigned long)jp.size(), &jw, &jh, &js, &jcs) != 0) continue;
        std::vector<unsigned char> dec((size_t)jw * jh * 3);
        if (tjDecompress2(tj, jp.data(), (unsigned long)jp.size(), dec.data(), jw, jw * 3, jh, TJPF_RGB, 0) != 0)
            continue;

        memset(img.data(), PAD_GRAY, img.size());
        float sc = (float)in_w / jw, sc2 = (float)in_h / jh;
        if (sc2 < sc) sc = sc2;
        if (sc > 1.0f) sc = 1.0f;
        int dw = (int)(jw * sc), dh = (int)(jh * sc);
        int ox = (in_w - dw) / 2, oy = (in_h - dh) / 2;
        for (int y = 0; y < dh; y++) {
            int sy = (int)(y / sc);
            if (sy >= jh) sy = jh - 1;
            unsigned char *o = img.data() + ((size_t)(y + oy) * in_w + ox) * 3;
            const unsigned char *srow = dec.data() + (size_t)sy * jw * 3;
            for (int x = 0; x < dw; x++) {
                int sx = (int)(x / sc);
                if (sx >= jw) sx = jw - 1;
                o[0] = srow[sx * 3];
                o[1] = srow[sx * 3 + 1];
                o[2] = srow[sx * 3 + 2];
                o += 3;
            }
        }

        in.buf = img.data();
        rknn_inputs_set(ctx, 1, &in);
        double e0 = now_ms();
        if (rknn_run(ctx, NULL) != RKNN_SUCC) {
            printf("rknn_run failed on %s\n", names[k].c_str());
            break;
        }
        sum_run += now_ms() - e0;
        rknn_outputs_get(ctx, nout, outs.data(), NULL);
        detect_result_group_t group;
        memset(&group, 0, sizeof(group));
        post_process((int8_t *)outs[0].buf, (int8_t *)outs[1].buf, (int8_t *)outs[2].buf, in_h, in_w, 0.35f, 0.45f,
                     1.0f, 1.0f, zps, scales, &group);
        rknn_outputs_release(ctx, nout, outs.data());

        int pc = 0;
        float best = 0.0f;
        const char *top = "-";
        for (int i = 0; i < group.count; i++) {
            if (group.results[i].prop > best) {
                best = group.results[i].prop;
                top = group.results[i].name;
            }
            if (strcmp(group.results[i].name, "person") == 0) pc++;
        }
        ok++;
        ndet += group.count;
        nperson += pc;
        if (pc > 0) {
            nwith++;
            float bestp = 0.0f;
            for (int i = 0; i < group.count; i++)
                if (strcmp(group.results[i].name, "person") == 0 && group.results[i].prop > bestp)
                    bestp = group.results[i].prop;
            sum_prop += bestp;
            nprop++;
        }
        printf("[%3zu] det=%d person=%d\n", k + 1, group.count, pc);
        if (csv) fprintf(csv, "%zu,%d,%d,%.4f,%s\n", k + 1, group.count, pc, pc ? best : 0.0f, top);
    }

    if (csv) fclose(csv);
    tjDestroy(tj);
    printf("eval: %d frames, det total=%d, person dets=%d, frames with person=%d, mean best person prop=%.1f%%, "
           "avg rknn_run=%.2f ms\n",
           ok, ndet, nperson, nwith, nprop ? 100.0 * sum_prop / nprop : 0.0, ok ? sum_run / ok : 0.0);
    printf("csv: %s\n", csv_path);
    return 0;
}

static int srv_open(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int on = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((unsigned short)port);
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) < 0) {
        printf("bind %d failed: %s\n", port, strerror(errno));
        close(fd);
        return -1;
    }
    listen(fd, 4);
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
    printf("MJPEG server listening on port %d, path /stream\n", port);
    return fd;
}

struct fbd {
    int fd = -1;
    unsigned char *mem = nullptr;
    int w = 0, h = 0, bpp = 0, stride = 0;
    unsigned int roff = 16, goff = 8, boff = 0;
    unsigned int amask = 0xFF000000u;
    size_t size = 0;
};

static int fb_open(struct fbd *f, const char *path) {
    f->fd = open(path, O_RDWR);
    if (f->fd < 0) {
        printf("open %s failed: %s\n", path, strerror(errno));
        return -1;
    }
    struct fb_var_screeninfo vi;
    struct fb_fix_screeninfo fi;
    if (ioctl(f->fd, FBIOGET_VSCREENINFO, &vi) < 0 || ioctl(f->fd, FBIOGET_FSCREENINFO, &fi) < 0) {
        printf("FBIOGET_*SCREENINFO failed\n");
        close(f->fd);
        f->fd = -1;
        return -1;
    }
    f->w = (int)vi.xres;
    f->h = (int)vi.yres;
    f->bpp = (int)vi.bits_per_pixel;
    f->stride = (int)fi.line_length;
    f->roff = vi.red.offset;
    f->goff = vi.green.offset;
    f->boff = vi.blue.offset;
    f->amask = vi.transp.length ? (0xFFu << vi.transp.offset) : 0xFF000000u;
    f->size = (size_t)f->stride * vi.yres_virtual;
    f->mem = (unsigned char *)mmap(NULL, f->size, PROT_READ | PROT_WRITE, MAP_SHARED, f->fd, 0);
    if (f->mem == MAP_FAILED) {
        printf("fb mmap failed: %s\n", strerror(errno));
        f->mem = nullptr;
        close(f->fd);
        f->fd = -1;
        return -1;
    }
    printf("framebuffer %s %dx%d %dbpp stride=%d rgba=%u,%u,%u,%u amask=0x%08x\n", path, f->w, f->h, f->bpp,
           f->stride, f->roff, f->goff, f->boff, vi.transp.offset, f->amask);
    return 0;
}

static void fb_close(struct fbd *f) {
    if (f->mem) munmap(f->mem, f->size);
    if (f->fd >= 0) close(f->fd);
    f->mem = nullptr;
    f->fd = -1;
}

// Nearest-neighbour scale of an RGB888 frame, centred on the panel as XRGB8888.
static void fb_show(struct fbd *f, const unsigned char *rgb, int sw, int sh) {
    if (!f->mem || f->bpp != 32) return;
    int dw = f->w;
    int dh = sh * dw / sw;
    if (dh > f->h) {
        dh = f->h;
        dw = sw * dh / sh;
    }
    int x0 = (f->w - dw) / 2;
    int y0 = (f->h - dh) / 2;
    if (y0 > 0) memset(f->mem, 0, (size_t)f->stride * y0);
    int sy = 0, yerr = 0;
    for (int y = 0; y < dh; y++) {
        const unsigned char *src = rgb + (size_t)sy * sw * 3;
        unsigned char *row = f->mem + (size_t)(y0 + y) * f->stride + (size_t)x0 * 4;
        unsigned int *dst = (unsigned int *)row;
        int sx = 0, xerr = 0;
        for (int x = 0; x < dw; x++) {
            const unsigned char *p = src + sx * 3;
            dst[x] = f->amask | ((unsigned int)p[0] << f->roff) | ((unsigned int)p[1] << f->goff) |
                     ((unsigned int)p[2] << f->boff);
            xerr += sw;
            while (xerr >= dw) {
                xerr -= dw;
                sx++;
            }
        }
        yerr += sh;
        while (yerr >= dh) {
            yerr -= dh;
            sy++;
        }
    }
    if (y0 + dh < f->h) {
        memset(f->mem + (size_t)(y0 + dh) * f->stride, 0, (size_t)f->stride * (f->h - y0 - dh));
    }
}

static int sock_queued(int fd) {
    int q = 0;
    if (ioctl(fd, TIOCOUTQ, &q) < 0) return 0;
    return q;
}

// Returns false only on a hard error. EAGAIN just leaves the rest in obuf for the next loop.
static bool flush_out(int fd, std::vector<unsigned char> &obuf, size_t &opos) {
    while (opos < obuf.size()) {
        ssize_t n = send(fd, obuf.data() + opos, obuf.size() - opos, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n > 0) {
            opos += (size_t)n;
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return true;
        return false;
    }
    obuf.clear();
    opos = 0;
    return true;
}

static int serve_http(int port, const char *root) {
    int fd = srv_open(port);
    if (fd < 0) return 1;
    printf("file server root=%s port=%d\n", root, port);
    while (!g_stop) {
        fd_set rf;
        FD_ZERO(&rf);
        FD_SET(fd, &rf);
        struct timeval stv = {1, 0};
        if (select(fd + 1, &rf, NULL, NULL, &stv) <= 0) continue;
        struct sockaddr_in peer;
        socklen_t plen = sizeof(peer);
        int c = accept(fd, (struct sockaddr *)&peer, &plen);
        if (c < 0) continue;
        struct timeval tv = {5, 0};
        setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        char req[1024];
        int n = read(c, req, sizeof(req) - 1);
        if (n > 0) {
            req[n] = 0;
            char path[512];
            if (sscanf(req, "GET %511s", path) == 1) {
                const char *p = path;
                if (*p == '/') p++;
                if (*p && !strstr(p, "..") && !strchr(p, '?')) {
                    char full[768];
                    snprintf(full, sizeof(full), "%s/%s", root, p);
                    FILE *f = fopen(full, "rb");
                    if (f) {
                        fseek(f, 0, SEEK_END);
                        long sz = ftell(f);
                        fseek(f, 0, SEEK_SET);
                        char hdr[256];
                        int hl = snprintf(hdr, sizeof(hdr),
                                          "HTTP/1.0 200 OK\r\nContent-Length: %ld\r\n"
                                          "Content-Type: application/octet-stream\r\n\r\n",
                                          sz);
                        write(c, hdr, (size_t)hl);
                        char buf[65536];
                        size_t r;
                        while ((r = fread(buf, 1, sizeof(buf), f)) > 0) {
                            size_t off = 0;
                            while (off < r) {
                                ssize_t w = write(c, buf + off, r - off);
                                if (w <= 0) break;
                                off += (size_t)w;
                            }
                            if (off < r) break;
                        }
                        fclose(f);
                    } else {
                        const char *nf = "HTTP/1.0 404 Not Found\r\nContent-Length: 0\r\n\r\n";
                        write(c, nf, strlen(nf));
                    }
                }
            }
        }
        close(c);
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *dev = (argc > 1) ? argv[1] : "/dev/video18";
    const char *model_path = (argc > 2) ? argv[2] : "model/yolov5s.rknn";
    int frames = (argc > 3) ? atoi(argv[3]) : 0;
    int port = (argc > 4) ? atoi(argv[4]) : 0;
    const char *fb_path = (argc > 5) ? argv[5] : "/dev/fb0";
    bool fb_swap = (argc > 6) && strcmp(argv[6], "swap") == 0;
    bool dump_on = false;
    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "dump") == 0) dump_on = true;
    }
    if (dump_on) mkdir("calib", 0755);
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, on_sigint);
    if (argc > 1 && strcmp(argv[1], "serve") == 0) {
        int sport = (argc > 2) ? atoi(argv[2]) : 8090;
        const char *root = (argc > 3) ? argv[3] : ".";
        return serve_http(sport, root);
    }

    FILE *f = fopen(model_path, "rb");
    if (!f) {
        printf("open %s failed\n", model_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t model_size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> model(model_size);
    if (fread(model.data(), 1, model_size, f) != model_size) {
        printf("read model failed\n");
        return 1;
    }
    fclose(f);

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model.data(), (uint32_t)model_size, 0, NULL);
    if (ret != RKNN_SUCC) {
        printf("rknn_init failed ret=%d\n", ret);
        return 1;
    }

    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    int in_w = in_attr.dims[2], in_h = in_attr.dims[1], in_c = in_attr.dims[3];
    if (in_attr.fmt == RKNN_TENSOR_NCHW) {
        in_c = in_attr.dims[1];
        in_h = in_attr.dims[2];
        in_w = in_attr.dims[3];
    }
    printf("model input %dx%dx%d, %d outputs\n", in_w, in_h, in_c, io_num.n_output);

    std::vector<rknn_tensor_attr> out_attrs(io_num.n_output);
    std::vector<int32_t> zps;
    std::vector<float> scales;
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        memset(&out_attrs[i], 0, sizeof(rknn_tensor_attr));
        out_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(rknn_tensor_attr));
        zps.push_back(out_attrs[i].zp);
        scales.push_back(out_attrs[i].scale);
    }

    printf("input attr: type=%d qnt=%d zp=%d scale=%.6f\n", in_attr.type, in_attr.qnt_type, in_attr.zp,
           in_attr.scale);
    for (uint32_t i = 0; i < io_num.n_output; i++)
        printf("out %u: type=%d qnt=%d zp=%d scale=%.6f dims=%d,%d,%d,%d\n", i, out_attrs[i].type,
               out_attrs[i].qnt_type, out_attrs[i].zp, out_attrs[i].scale, out_attrs[i].dims[0],
               out_attrs[i].dims[1], out_attrs[i].dims[2], out_attrs[i].dims[3]);

    if (argc > 1 && strcmp(argv[1], "eval") == 0) {
        const char *dir = (argc > 3) ? argv[3] : "calib";
        int maxf = (argc > 4) ? atoi(argv[4]) : 0;
        int rc = eval_dir(ctx, model_path, dir, in_w, in_h, zps, scales, io_num.n_output, maxf);
        rknn_destroy(ctx);
        return rc;
    }

    struct cam c;
    if (cam_open(&c, dev, 640, 480, 30) < 0) return 1;

    struct fbd fb;
    bool fb_on = strcmp(fb_path, "none") != 0 && fb_open(&fb, fb_path) == 0;
    if (fb_on && fb_swap) {
        unsigned int t = fb.roff;
        fb.roff = fb.boff;
        fb.boff = t;
        printf("fb channels swapped: rgba=%u,%u,%u,%u amask=0x%08x\n", fb.roff, fb.goff, fb.boff,
               (fb.amask == 0xFF000000u) ? 24u : (unsigned)(__builtin_ctz(fb.amask)), fb.amask);
    }

    int srv_fd = -1;
    if (port > 0) {
        srv_fd = srv_open(port);
        if (srv_fd < 0) return 1;
    }
    // Letterbox padding is not worth encoding: stream only the real picture rows.
    const int crop_y = (in_h > c.h) ? (in_h - c.h) / 2 : 0;
    const int crop_h = (in_h > c.h) ? c.h : in_h;

    struct frame_slot {
        std::vector<unsigned char> rgb;
        int state = 0;  // 0 free, 1 ready for main, 2 owned by main, 3 queued for encoder
    };
    struct pipeline {
        std::mutex m;
        std::condition_variable cv_ready, cv_enc;
        frame_slot slots[3];
        int latest = -1;
        std::vector<int> enc_q;
        std::atomic<bool> stop{false};
        std::atomic<int> cam_drops{0};
        std::atomic<int> net_drops{0};
        std::atomic<int> cli{-1};
    } p;
    for (int i = 0; i < 3; i++) p.slots[i].rgb.resize((size_t)in_w * in_h * in_c);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = (uint32_t)((size_t)in_w * in_h * in_c);
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = nullptr;

    std::vector<rknn_output> outputs(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++) outputs[i].want_float = 0;

    stat_acc st_wait, st_conv, st_run, st_post, st_jpeg, st_fb, st_total;

    // Stage 1: capture + convert, publish only the newest frame into a free slot.
    std::thread cap_thr([&]() {
        int errs = 0;
        while (!p.stop) {
            int dropped = 0;
            size_t bytes = 0;
            int idx = cam_dq_latest(&c, &bytes, &dropped);
            if (idx < 0) {
                if (p.stop) break;
                if (++errs > 3) {
                    printf("camera stopped delivering frames\n");
                    p.stop = true;
                    p.cv_ready.notify_all();
                    break;
                }
                continue;
            }
            errs = 0;
            if (p.stop) {
                cam_q(&c, idx);
                break;
            }
            int s = -1;
            {
                std::lock_guard<std::mutex> lk(p.m);
                for (int i = 0; i < 3; i++) {
                    if (p.slots[i].state == 0) {
                        s = i;
                        p.slots[i].state = 2;
                        break;
                    }
                }
            }
            if (s < 0) {
                p.cam_drops++;
                cam_q(&c, idx);
                continue;
            }
            double d0 = now_ms();
            yuyv_to_rgb_letterbox((const unsigned char *)c.start[idx], c.w, c.h, p.slots[s].rgb.data(), in_w, in_h);
            cam_q(&c, idx);
            st_conv.add(now_ms() - d0);
            {
                std::lock_guard<std::mutex> lk(p.m);
                if (p.latest >= 0) {
                    p.slots[p.latest].state = 0;
                    p.cam_drops++;
                }
                p.latest = s;
                p.slots[s].state = 1;
            }
            p.cv_ready.notify_one();
        }
    });

    // Stage 3: JPEG + non-blocking send + client handling, so a slow link never stalls the NPU.
    std::thread enc_thr([&]() {
        tjhandle tj = nullptr;
        std::vector<unsigned char> obuf;
        std::vector<unsigned char> jpeg((size_t)in_w * in_h * 3 + 4096);
        size_t opos = 0;
        bool need_hdr = false;
        double pend_since = 0;
        int cli_fd = -1;

        auto drop_client = [&](const char *why) {
            close(cli_fd);
            cli_fd = -1;
            p.cli = -1;
            obuf.clear();
            opos = 0;
            pend_since = 0;
            printf("%s\n", why);
        };

        while (true) {
            int s = -1;
            {
                std::unique_lock<std::mutex> lk(p.m);
                p.cv_enc.wait_for(lk, std::chrono::milliseconds(100), [&]() { return !p.enc_q.empty() || p.stop; });
                if (!p.enc_q.empty()) {
                    s = p.enc_q.front();
                    p.enc_q.erase(p.enc_q.begin());
                } else if (p.stop) {
                    break;
                }
            }

            if (srv_fd >= 0) {
                struct sockaddr_in peer;
                socklen_t plen = sizeof(peer);
                int fd = accept(srv_fd, (struct sockaddr *)&peer, &plen);
                if (fd >= 0) {
                    if (cli_fd >= 0) {
                        close(cli_fd);
                        printf("replaced stale client\n");
                    }
                    cli_fd = fd;
                    int nodelay = 1;
                    setsockopt(cli_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
                    char ip[32];
                    inet_ntop(AF_INET, &peer.sin_addr, ip, sizeof(ip));
                    printf("client %s connected\n", ip);
                    obuf.clear();
                    opos = 0;
                    need_hdr = true;
                    pend_since = 0;
                    p.cli = cli_fd;
                }
            }
            if (s < 0) continue;

            if (fb_on) {
                double d0 = now_ms();
                fb_show(&fb, p.slots[s].rgb.data() + (size_t)crop_y * in_w * 3, in_w, crop_h);
                st_fb.add(now_ms() - d0);
            }

            if (cli_fd >= 0) {
                double j0 = now_ms();
                if (!obuf.empty()) {
                    if (!flush_out(cli_fd, obuf, opos)) {
                        drop_client("client gone");
                    } else if (obuf.empty()) {
                        pend_since = 0;
                    } else if (pend_since == 0) {
                        pend_since = j0;
                    } else if (j0 - pend_since > 2000) {
                        drop_client("client stalled >2s, dropped");
                    } else {
                        p.net_drops++;
                    }
                } else {
                    if (!tj) tj = tjInitCompress();
                    unsigned char *jp = jpeg.data();
                    unsigned long jsz = (unsigned long)jpeg.size();
                    const unsigned char *src = p.slots[s].rgb.data() + (size_t)crop_y * in_w * 3;
                    if (tj && tjCompress2(tj, src, in_w, in_w * 3, crop_h, TJPF_RGB, &jp, &jsz, TJSAMP_420, 60, 0) == 0) {
                        char part[128];
                        int n = snprintf(part, sizeof(part),
                                         "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %lu\r\n\r\n", jsz);
                        obuf.clear();
                        if (need_hdr) {
                            static const char HDR[] = "HTTP/1.0 200 OK\r\n"
                                                      "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                                                      "Cache-Control: no-cache\r\n"
                                                      "Connection: close\r\n\r\n";
                            obuf.insert(obuf.end(), HDR, HDR + sizeof(HDR) - 1);
                            need_hdr = false;
                        }
                        obuf.insert(obuf.end(), part, part + n);
                        obuf.insert(obuf.end(), jp, jp + jsz);
                        obuf.push_back('\r');
                        obuf.push_back('\n');
                        opos = 0;
                        if (!flush_out(cli_fd, obuf, opos)) {
                            drop_client("client gone");
                        } else if (!obuf.empty()) {
                            pend_since = now_ms();
                        }
                    }
                }
                st_jpeg.add(now_ms() - j0);
            }

            {
                std::lock_guard<std::mutex> lk(p.m);
                p.slots[s].state = 0;
            }
        }
        if (cli_fd >= 0) close(cli_fd);
        if (tj) tjDestroy(tj);
    });

    double t_begin = now_ms();
    int done = 0, det_total = 0;
    int last_count = 0;
    char last_top[64] = "-";
    person_track tracks[32];
    memset(tracks, 0, sizeof(tracks));
    int person_now = 0, person_total = 0, person_det = 0;
    int dumped = 0;
    double last_count_ms = -1e9;
    float last_count_cx = 0, last_count_cy = 0;

    while ((frames <= 0 || done < frames) && !g_stop) {
        double t0 = now_ms();
        int s = -1;
        {
            std::unique_lock<std::mutex> lk(p.m);
            p.cv_ready.wait(lk, [&]() { return p.latest >= 0 || p.stop; });
            if (p.latest < 0) break;
            s = p.latest;
            p.latest = -1;
            p.slots[s].state = 2;
        }
        double t1 = now_ms();
        st_wait.add(t1 - t0);

        unsigned char *rgb = p.slots[s].rgb.data();
        if (dump_on && dumped < (frames > 0 ? frames : 1000)) {
            char path[64];
            snprintf(path, sizeof(path), "calib/frame_%04d.jpg", dumped);
            if (save_jpeg(rgb, in_w, in_h, path) == 0) dumped++;
        }
        inputs[0].buf = rgb;
        rknn_inputs_set(ctx, io_num.n_input, inputs);
        double t3 = now_ms();
        ret = rknn_run(ctx, NULL);
        double t4 = now_ms();
        if (ret != RKNN_SUCC) {
            printf("rknn_run failed ret=%d\n", ret);
            break;
        }
        st_run.add(t4 - t3);

        rknn_outputs_get(ctx, io_num.n_output, outputs.data(), NULL);
        detect_result_group_t group;
        memset(&group, 0, sizeof(group));
        post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, in_h, in_w, 0.35f,
                     0.45f, 1.0f, 1.0f, zps, scales, &group);
        double t5 = now_ms();
        st_post.add(t5 - t4);
        rknn_outputs_release(ctx, io_num.n_output, outputs.data());

        last_count = group.count;
        det_total += group.count;
        if (group.count > 0) {
            const detect_result_t *best = &group.results[0];
            for (int i = 1; i < group.count; i++) {
                if (group.results[i].prop > best->prop) best = &group.results[i];
            }
            snprintf(last_top, sizeof(last_top), "%s %.0f%%", best->name, best->prop * 100.0f);
        } else {
            snprintf(last_top, sizeof(last_top), "-");
        }

        for (int t = 0; t < 32; t++) {
            if (tracks[t].active) {
                tracks[t].misses++;
                tracks[t].age++;
                tracks[t].seen = false;
            }
        }
        for (int i = 0; i < group.count; i++) {
            const detect_result_t *d = &group.results[i];
            if (strcmp(d->name, "person") != 0) continue;
            person_det++;
            int bi = -1;
            float biou = 0.1f;
            float dcx = (d->box.left + d->box.right) * 0.5f;
            float dcy = (d->box.top + d->box.bottom) * 0.5f;
            for (int t = 0; t < 32; t++) {
                if (!tracks[t].active) continue;
                float iou =
                    box_iou(tracks[t].x0, tracks[t].y0, tracks[t].x1, tracks[t].y1, d->box.left, d->box.top,
                            d->box.right, d->box.bottom);
                if (iou > biou) {
                    biou = iou;
                    bi = t;
                }
            }
            if (bi < 0) {
                for (int t = 0; t < 32; t++) {
                    if (!tracks[t].active) continue;
                    float tx = (tracks[t].x0 + tracks[t].x1) * 0.5f;
                    float ty = (tracks[t].y0 + tracks[t].y1) * 0.5f;
                    float dx = dcx - tx, dy = dcy - ty;
                    if (dx * dx + dy * dy < 90.0f * 90.0f) {
                        bi = t;
                        break;
                    }
                }
            }
            if (bi < 0) {
                for (int t = 0; t < 32; t++) {
                    if (!tracks[t].active) {
                        bi = t;
                        break;
                    }
                }
                if (bi < 0) continue;
                tracks[bi].hits = 0;
                tracks[bi].age = 0;
                tracks[bi].counted = false;
                tracks[bi].active = true;
            }
            person_track *tr = &tracks[bi];
            tr->x0 = d->box.left;
            tr->y0 = d->box.top;
            tr->x1 = d->box.right;
            tr->y1 = d->box.bottom;
            tr->misses = 0;
            tr->seen = true;
            if (tr->hits < 1000) tr->hits++;
            if (tr->hits >= 5 && !tr->counted) {
                tr->counted = true;
                bool dup = false;
                for (int t = 0; t < 32; t++) {
                    if (t == bi || !tracks[t].active || !tracks[t].counted) continue;
                    if (box_iou(tracks[t].x0, tracks[t].y0, tracks[t].x1, tracks[t].y1, d->box.left, d->box.top,
                                d->box.right, d->box.bottom) > 0.1f) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) {
                    float ccx = (d->box.left + d->box.right) * 0.5f;
                    float ccy = (d->box.top + d->box.bottom) * 0.5f;
                    double nowc = now_ms();
                    float dx = ccx - last_count_cx, dy = ccy - last_count_cy;
                    bool same = nowc - last_count_ms < 3000.0 && dx * dx + dy * dy < 200.0f * 200.0f;
                    if (!same) {
                        person_total++;
                        last_count_ms = nowc;
                        last_count_cx = ccx;
                        last_count_cy = ccy;
                        log_person(person_total);
                    }
                }
            }
        }
        person_now = 0;
        for (int t = 0; t < 32; t++) {
            if (!tracks[t].active) continue;
            if (tracks[t].seen && tracks[t].hits >= 2) person_now++;
            if (tracks[t].misses > 30) tracks[t].active = false;
            else if (tracks[t].age > 90 && !tracks[t].counted) tracks[t].active = false;
        }

        if (fb_on || p.cli >= 0) {
            for (int i = 0; i < group.count; i++) {
                const detect_result_t *d = &group.results[i];
                const unsigned char *col = PALETTE[abs(d->box.left + d->box.top) % 10];
                draw_rect(rgb, in_w, in_h, d->box.left, d->box.top, d->box.right, d->box.bottom, col, 3);
                char lbl[64];
                snprintf(lbl, sizeof(lbl), "%s %d%%", d->name, (int)(d->prop * 100.0f + 0.5f));
                int ly = d->box.top - 8 * 2 - 2;
                if (ly < 0) ly = d->box.top + 2;
                draw_text(rgb, in_w, in_h, d->box.left, ly, lbl, col, 2);
            }
            char hud[64];
            snprintf(hud, sizeof(hud), "person %d total %d", person_now, person_total);
            draw_text(rgb, in_w, in_h, 8, 8, hud, PALETTE[9], 2);
        }
        {
            std::lock_guard<std::mutex> lk(p.m);
            if (p.enc_q.size() < 3) {
                p.enc_q.push_back(s);
                p.slots[s].state = 3;
            } else {
                p.slots[s].state = 0;
                p.net_drops++;
            }
        }
        p.cv_enc.notify_one();

        double t6 = now_ms();
        st_total.add(t6 - t0);
        done++;

        if (frames > 0 && done % 25 == 0) {
            printf("[%3d] det=%d person=%d/%d top=%s\n", done, last_count, person_now, person_total, last_top);
        } else if (frames <= 0 && done % 100 == 0) {
            printf("[%4d] %.1f fps det=%d person=%d/%d top=%s q=%d drop=%d/%d\n", done,
                   1000.0 * done / (t6 - t_begin), last_count, person_now, person_total, last_top,
                   p.cli >= 0 ? sock_queued(p.cli.load()) : 0, p.cam_drops.load(), p.net_drops.load());
        }
    }

    p.stop = true;
    p.cv_ready.notify_all();
    p.cv_enc.notify_all();
    cap_thr.join();
    enc_thr.join();

    double elapsed = now_ms() - t_begin;
    printf("\nframes=%d elapsed=%.0f ms -> %.2f fps, avg det=%.2f, last det=%d top=%s, person det=%d total=%d\n", done,
           elapsed, 1000.0 * done / elapsed, done ? (double)det_total / done : 0.0, last_count, last_top,
           person_det, person_total);
    printf("dropped: camera=%d net=%d\n", p.cam_drops.load(), p.net_drops.load());
    if (dump_on) printf("dumped %d frames to ./calib/\n", dumped);
    printf("per-stage ms (avg/min/max, p50/p95):\n");
    printf("  wait   %6.2f / %6.2f / %6.2f  p50 %6.2f p95 %6.2f\n", st_wait.avg(), st_wait.mn, st_wait.mx,
           st_wait.pct(0.5), st_wait.pct(0.95));
    printf("  conv   %6.2f / %6.2f / %6.2f  p50 %6.2f p95 %6.2f\n", st_conv.avg(), st_conv.mn, st_conv.mx,
           st_conv.pct(0.5), st_conv.pct(0.95));
    printf("  infer  %6.2f / %6.2f / %6.2f  p50 %6.2f p95 %6.2f\n", st_run.avg(), st_run.mn, st_run.mx,
           st_run.pct(0.5), st_run.pct(0.95));
    printf("  post   %6.2f / %6.2f / %6.2f  p50 %6.2f p95 %6.2f\n", st_post.avg(), st_post.mn, st_post.mx,
           st_post.pct(0.5), st_post.pct(0.95));
    printf("  jpeg   %6.2f / %6.2f / %6.2f  p50 %6.2f p95 %6.2f\n", st_jpeg.avg(), st_jpeg.mn, st_jpeg.mx,
           st_jpeg.pct(0.5), st_jpeg.pct(0.95));
    printf("  fb     %6.2f / %6.2f / %6.2f  p50 %6.2f p95 %6.2f\n", st_fb.avg(), st_fb.mn, st_fb.mx, st_fb.pct(0.5),
           st_fb.pct(0.95));
    printf("  total  %6.2f / %6.2f / %6.2f  p50 %6.2f p95 %6.2f\n", st_total.avg(), st_total.mn, st_total.mx,
           st_total.pct(0.5), st_total.pct(0.95));

    if (srv_fd >= 0) close(srv_fd);
    fb_close(&fb);
    cam_close(&c);
    rknn_destroy(ctx);
    deinitPostProcess();
    return 0;
}
