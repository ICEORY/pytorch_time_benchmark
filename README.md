## pytorch evaluate running time

reference: https://discuss.pytorch.org/t/measuring-gpu-tensor-operation-speed/2513/4

### Results

```
loop #0, batch GPU 6.479860 s
loop #1, batch GPU 0.000720 s
loop #2, batch GPU 0.000275 s
loop #3, batch GPU 0.000266 s
loop #4, batch GPU 0.000264 s
loop #5, batch GPU 0.000262 s
loop #6, batch GPU 0.000267 s
loop #7, batch GPU 0.000264 s
loop #8, batch GPU 0.000278 s
loop #9, batch GPU 0.000313 s
```

### Note

The first two loops cost more time than the other operations. We always run hundreds or thousands loops and use the 80%/90% smallest values to obtain a more stable result.

```
ngimel Apr '17
I believe cublas handles are allocated lazily now, which means that first operation requiring cublas will have an overhead of creating cublas handle, and that includes some internal allocations. So there’s no way to avoid it other than calling some function requiring cublas before the timing loop.
```