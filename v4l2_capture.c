// V4L2 capture smoke test: YUYV, mmap streaming, prints fps and dqbuf latency
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>

#define BUF_NUM 4

struct buf {
    void *start;
    size_t length;
};

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static int xioctl(int fd, unsigned long req, void *arg) {
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

int main(int argc, char **argv) {
    const char *dev = (argc > 1) ? argv[1] : "/dev/video18";
    int want_w = (argc > 2) ? atoi(argv[2]) : 640;
    int want_h = (argc > 3) ? atoi(argv[3]) : 480;
    int frames = (argc > 4) ? atoi(argv[4]) : 150;
    int want_fps = (argc > 5) ? atoi(argv[5]) : 30;

    int fd = open(dev, O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        printf("open %s failed: %s\n", dev, strerror(errno));
        return 1;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = want_w;
    fmt.fmt.pix.height = want_h;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        printf("VIDIOC_S_FMT failed: %s\n", strerror(errno));
        return 1;
    }
    printf("negotiated %ux%u fourcc=%c%c%c%c sizeimage=%u bytesperline=%u\n", fmt.fmt.pix.width, fmt.fmt.pix.height,
           fmt.fmt.pix.pixelformat & 0xff, (fmt.fmt.pix.pixelformat >> 8) & 0xff, (fmt.fmt.pix.pixelformat >> 16) & 0xff,
           (fmt.fmt.pix.pixelformat >> 24) & 0xff, fmt.fmt.pix.sizeimage, fmt.fmt.pix.bytesperline);

    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = want_fps;
    if (xioctl(fd, VIDIOC_S_PARM, &parm) < 0) {
        printf("VIDIOC_S_PARM %d fps failed: %s\n", want_fps, strerror(errno));
    }
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_G_PARM, &parm) == 0) {
        unsigned int num = parm.parm.capture.timeperframe.numerator;
        unsigned int den = parm.parm.capture.timeperframe.denominator;
        printf("requested %d fps, device reports %.2f fps (%u/%u)%s\n", want_fps, num ? (double)den / num : 0.0, num,
               den, (parm.parm.capture.capability & V4L2_CAP_TIMEPERFRAME) ? "" : " [no timeperframe support]");
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = BUF_NUM;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        printf("VIDIOC_REQBUFS failed: %s\n", strerror(errno));
        return 1;
    }
    printf("buffers=%u\n", req.count);

    struct buf bufs[BUF_NUM];
    memset(bufs, 0, sizeof(bufs));
    for (unsigned int i = 0; i < req.count; i++) {
        struct v4l2_buffer b;
        memset(&b, 0, sizeof(b));
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index = i;
        if (xioctl(fd, VIDIOC_QUERYBUF, &b) < 0) {
            printf("VIDIOC_QUERYBUF %u failed: %s\n", i, strerror(errno));
            return 1;
        }
        bufs[i].length = b.length;
        bufs[i].start = mmap(NULL, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, b.m.offset);
        if (bufs[i].start == MAP_FAILED) {
            printf("mmap %u failed: %s\n", i, strerror(errno));
            return 1;
        }
        if (xioctl(fd, VIDIOC_QBUF, &b) < 0) {
            printf("VIDIOC_QBUF %u failed: %s\n", i, strerror(errno));
            return 1;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        printf("VIDIOC_STREAMON failed: %s\n", strerror(errno));
        return 1;
    }

    double t_start = now_ms();
    double dq_sum = 0.0, dq_min = 1e9, dq_max = 0.0;
    double sel_sum = 0.0, last_t = 0.0, interval_sum = 0.0;
    int got = 0;
    while (got < frames) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        struct timeval tv;
        tv.tv_sec = 2;
        tv.tv_usec = 0;
        double ts = now_ms();
        int r = select(fd + 1, &fds, NULL, NULL, &tv);
        if (r > 0) sel_sum += now_ms() - ts;
        if (r < 0) {
            if (errno == EINTR) continue;
            printf("select failed: %s\n", strerror(errno));
            break;
        }
        if (r == 0) {
            printf("select timeout at frame %d\n", got);
            break;
        }

        double t0 = now_ms();
        struct v4l2_buffer b;
        memset(&b, 0, sizeof(b));
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd, VIDIOC_DQBUF, &b) < 0) {
            if (errno == EAGAIN) continue;
            printf("VIDIOC_DQBUF failed: %s\n", strerror(errno));
            break;
        }
        double dq = now_ms() - t0;
        dq_sum += dq;
        if (dq < dq_min) dq_min = dq;
        if (dq > dq_max) dq_max = dq;

        double tframe = now_ms();
        if (last_t > 0.0) interval_sum += tframe - last_t;
        last_t = tframe;

        if (got == 0) {
            FILE *f = fopen("/root/demo/frame.yuyv", "wb");
            if (f) {
                fwrite(bufs[b.index].start, 1, b.bytesused, f);
                fclose(f);
                printf("first frame: index=%u bytesused=%u saved to /root/demo/frame.yuyv\n", b.index, b.bytesused);
            }
        }

        if (xioctl(fd, VIDIOC_QBUF, &b) < 0) {
            printf("VIDIOC_QBUF failed: %s\n", strerror(errno));
            break;
        }
        got++;
    }

    double elapsed = now_ms() - t_start;
    if (got > 0) {
        printf("captured %d frames in %.1f ms -> %.2f fps\n", got, elapsed, 1000.0 * got / elapsed);
        printf("  frame interval avg=%.2f ms | select wait avg=%.2f ms | dqbuf avg=%.2f min=%.2f max=%.2f ms\n",
               (got > 1) ? interval_sum / (got - 1) : 0.0, sel_sum / got, dq_sum / got, dq_min, dq_max);
    }

    xioctl(fd, VIDIOC_STREAMOFF, &type);
    for (unsigned int i = 0; i < req.count; i++) {
        if (bufs[i].start && bufs[i].start != MAP_FAILED) munmap(bufs[i].start, bufs[i].length);
    }
    close(fd);
    return 0;
}
