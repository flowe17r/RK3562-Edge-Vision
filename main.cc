// RK3562 yolov5s RKNN demo: raw RGB input -> NPU -> postprocess -> print
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <vector>

#include "postprocess.h"
#include "rknn_api.h"

static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("fopen %s failed\n", path);
        exit(-1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc(n);
    if (!buf || fread(buf, 1, n, f) != (size_t)n) {
        printf("read %s failed\n", path);
        exit(-1);
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static double now_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void print_attr(const char *tag, rknn_tensor_attr *a) {
    printf("%s: name=%s dims=[", tag, a->name);
    for (uint32_t i = 0; i < a->n_dims; i++) {
        printf("%d%s", a->dims[i], (i + 1 < a->n_dims) ? "," : "");
    }
    printf("] fmt=%d type=%d zp=%d scale=%f size=%d\n", a->fmt, a->type, a->zp, a->scale, a->size);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: %s model.rknn input_%dx%dx3.rgb [loops]\n", argv[0], 640, 640);
        return -1;
    }
    int loops = (argc > 3) ? atoi(argv[3]) : 1;
    if (loops < 1) loops = 1;

    size_t model_size = 0, img_size = 0;
    unsigned char *model = read_file(argv[1], &model_size);
    unsigned char *img = read_file(argv[2], &img_size);

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model, (uint32_t)model_size, 0, NULL);
    if (ret != RKNN_SUCC) {
        printf("rknn_init failed ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version ver;
    memset(&ver, 0, sizeof(ver));
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == RKNN_SUCC) {
        printf("sdk api=%s driver=%s\n", ver.api_version, ver.drv_version);
    }

    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("query in/out num failed ret=%d\n", ret);
        return -1;
    }
    printf("n_input=%d n_output=%d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    print_attr("input", &in_attr);

    int w = in_attr.dims[2], h = in_attr.dims[1], c = in_attr.dims[3];
    if (in_attr.fmt == RKNN_TENSOR_NCHW) {
        c = in_attr.dims[1];
        h = in_attr.dims[2];
        w = in_attr.dims[3];
    }
    size_t expect = (size_t)w * h * c;
    printf("model input %dx%dx%d expect %zu bytes, file has %zu\n", w, h, c, expect, img_size);
    if (img_size != expect) {
        printf("input size mismatch, abort\n");
        return -1;
    }

    std::vector<rknn_tensor_attr> out_attrs(io_num.n_output);
    std::vector<int32_t> zps;
    std::vector<float> scales;
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        memset(&out_attrs[i], 0, sizeof(rknn_tensor_attr));
        out_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(rknn_tensor_attr));
        print_attr("output", &out_attrs[i]);
        zps.push_back(out_attrs[i].zp);
        scales.push_back(out_attrs[i].scale);
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = (uint32_t)expect;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = img;

    std::vector<rknn_output> outputs(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++) outputs[i].want_float = 0;

    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret != RKNN_SUCC) {
        printf("rknn_inputs_set failed ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(ctx, NULL);
    if (ret != RKNN_SUCC) {
        printf("rknn_run failed ret=%d\n", ret);
        return -1;
    }
    rknn_outputs_get(ctx, io_num.n_output, outputs.data(), NULL);
    rknn_outputs_release(ctx, io_num.n_output, outputs.data());

    double total = 0.0, best = 1e9, worst = 0.0;
    for (int i = 0; i < loops; i++) {
        double t0 = now_ms();
        ret = rknn_run(ctx, NULL);
        double t1 = now_ms();
        if (ret != RKNN_SUCC) {
            printf("rknn_run failed ret=%d\n", ret);
            return -1;
        }
        double ms = t1 - t0;
        total += ms;
        if (ms < best) best = ms;
        if (ms > worst) worst = ms;

        rknn_outputs_get(ctx, io_num.n_output, outputs.data(), NULL);
        if (i == 0) {
            detect_result_group_t group;
            memset(&group, 0, sizeof(group));
            post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, h, w,
                         0.25f, 0.45f, 1.0f, 1.0f, zps, scales, &group);
            printf("detections=%d\n", group.count);
            for (int j = 0; j < group.count; j++) {
                detect_result_t *d = &group.results[j];
                printf("  %-12s %5.1f%%  [%d,%d,%d,%d]\n", d->name, d->prop * 100.0f, d->box.left, d->box.top,
                       d->box.right, d->box.bottom);
            }
        }
        rknn_outputs_release(ctx, io_num.n_output, outputs.data());
    }

    printf("npu run ms: avg=%.2f min=%.2f max=%.2f over %d loops (%.1f fps)\n", total / loops, best, worst, loops,
           1000.0 * loops / total);

    rknn_destroy(ctx);
    deinitPostProcess();
    free(model);
    free(img);
    return 0;
}
