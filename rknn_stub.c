// Link-only stub for cross compiling: the board provides the real librknnrt.so
// at runtime. Signatures intentionally left unprototyped; only symbols matter.
int rknn_init() { return -1; }
int rknn_destroy() { return -1; }
int rknn_query() { return -1; }
int rknn_inputs_set() { return -1; }
int rknn_outputs_get() { return -1; }
int rknn_outputs_release() { return -1; }
int rknn_run() { return -1; }
