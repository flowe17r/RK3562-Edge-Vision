// Link-only stubs for TurboJPEG; the board provides libturbojpeg.so.0 at runtime.
unsigned long tjInitCompress() { return 0; }
int tjCompress2() { return -1; }
int tjDestroy() { return -1; }
int tjFree() { return 0; }
unsigned long tjInitDecompress() { return 0; }
int tjDecompressHeader3() { return -1; }
int tjDecompress2() { return -1; }
