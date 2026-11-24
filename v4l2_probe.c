// Enumerate /dev/videoN formats, frame sizes and frame rates (v4l2-ctl --list-formats-ext 的替代)
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

static void fourcc(unsigned int f, char *out) {
    out[0] = f & 0xff;
    out[1] = (f >> 8) & 0xff;
    out[2] = (f >> 16) & 0xff;
    out[3] = (f >> 24) & 0xff;
    out[4] = 0;
}

int main(int argc, char **argv) {
    const char *dev = (argc > 1) ? argv[1] : "/dev/video18";
    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        printf("open %s failed: %s\n", dev, strerror(errno));
        return 1;
    }

    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
        printf("VIDIOC_QUERYCAP failed: %s\n", strerror(errno));
        return 1;
    }
    printf("%s\ndriver=%s card=%s bus=%s\ncaps=0x%08x%s%s%s%s\n", dev, cap.driver, cap.card, cap.bus_info,
           cap.capabilities, (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ? " CAPTURE" : "",
           (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) ? " CAPTURE_MPLANE" : "",
           (cap.capabilities & V4L2_CAP_STREAMING) ? " STREAMING" : "",
           (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? " DEVICE_CAPS" : "");

    unsigned int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) && (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE)) {
        type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    }
    printf("using type=%s\n", type == V4L2_BUF_TYPE_VIDEO_CAPTURE ? "VIDEO_CAPTURE" : "VIDEO_CAPTURE_MPLANE");

    struct v4l2_fmtdesc f;
    memset(&f, 0, sizeof(f));
    f.type = type;
    for (f.index = 0; ioctl(fd, VIDIOC_ENUM_FMT, &f) == 0; f.index++) {
        char cc[5];
        fourcc(f.pixelformat, cc);
        printf("fmt[%u] %s (%s)\n", f.index, cc, f.description);

        struct v4l2_frmsizeenum fs;
        memset(&fs, 0, sizeof(fs));
        fs.pixel_format = f.pixelformat;
        for (fs.index = 0; ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fs) == 0; fs.index++) {
            if (fs.type != V4L2_FRMSIZE_TYPE_DISCRETE) {
                printf("   stepwise (not enumerated)\n");
                break;
            }
            printf("   %ux%u :", fs.discrete.width, fs.discrete.height);

            struct v4l2_frmivalenum fi;
            memset(&fi, 0, sizeof(fi));
            fi.pixel_format = f.pixelformat;
            fi.width = fs.discrete.width;
            fi.height = fs.discrete.height;
            int shown = 0;
            for (fi.index = 0; ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &fi) == 0; fi.index++) {
                if (fi.type != V4L2_FRMIVAL_TYPE_DISCRETE) break;
                if (fi.discrete.numerator)
                    printf(" %.1ffps", (double)fi.discrete.denominator / (double)fi.discrete.numerator);
                else
                    printf(" ?fps");
                shown = 1;
            }
            if (!shown) printf(" (no interval info)");
            printf("\n");
        }
    }

    close(fd);
    return 0;
}
