/* Portable C kernels for samebit. Only fixed-width integer arithmetic is
 * used. Signed right shift and signed division are implementation-defined
 * or truncating in C, so floor shifts are written out explicitly. */

#include <stdint.h>

static int64_t floor_shift(int64_t v, int s) {
  if (s == 0) return v;
  if (v >= 0) return v >> s;
  return -(int64_t)(((uint64_t)(-(v + 1)) >> s) + 1);
}

static int64_t rshift_round(int64_t v, int s) {
  return s == 0 ? v : floor_shift(v + ((int64_t)1 << (s - 1)), s);
}

static int8_t sat8(int64_t v) {
  return (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
}

static int64_t isqrt32(int64_t n) {
  int64_t res = 0, bit = (int64_t)1 << 30;
  for (int i = 0; i < 16; i++) {
    if (n >= res + bit) {
      n -= res + bit;
      res = (res >> 1) + bit;
    } else {
      res >>= 1;
    }
    bit >>= 2;
  }
  return res;
}

/* order 0: k ascending, 1: k descending, 2: blocks of 7 visited back to
 * front. The spec's bounds make every order give the same int32 result. */
void sb_matmul(const int32_t *a, const int32_t *b, int32_t *out, int m, int k,
               int n, int order) {
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int32_t acc = 0;
      if (order == 0) {
        for (int t = 0; t < k; t++) acc += a[i * k + t] * b[t * n + j];
      } else if (order == 1) {
        for (int t = k - 1; t >= 0; t--) acc += a[i * k + t] * b[t * n + j];
      } else {
        for (int start = ((k - 1) / 7) * 7; start >= 0; start -= 7)
          for (int t = start; t < start + 7 && t < k; t++)
            acc += a[i * k + t] * b[t * n + j];
      }
      out[i * n + j] = acc;
    }
  }
}

void sb_requantize(const int64_t *x, const int64_t *m, const int64_t *s,
                   int8_t *out, int rows, int cols, int per_channel) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) {
      int c = per_channel ? j : 0;
      out[i * cols + j] = sat8(rshift_round(x[i * cols + j] * m[c], (int)s[c]));
    }
}

void sb_rmsnorm(const int64_t *x, const int64_t *g, const int64_t *m,
                const int64_t *s, int8_t *out, int rows, int d,
                int per_channel) {
  for (int i = 0; i < rows; i++) {
    const int64_t *row = x + (int64_t)i * d;
    int64_t sq = 0;
    for (int j = 0; j < d; j++) sq += row[j] * row[j];
    int64_t r = isqrt32((sq << 16) / d); /* both operands non-negative */
    if (r < 1) r = 1;
    int64_t inv = ((int64_t)1 << 30) / r;
    for (int j = 0; j < d; j++) {
      int c = per_channel ? j : 0;
      int64_t v = rshift_round(row[j] * inv, 14) * g[c];
      out[i * d + j] = sat8(rshift_round(v * m[c], (int)s[c]));
    }
  }
}

void sb_rope(const int64_t *x, const int64_t *cs, const int64_t *sn,
             int8_t *out, int t, int h, int d) {
  int half = d / 2;
  for (int i = 0; i < t; i++)
    for (int k = 0; k < h; k++) {
      const int64_t *v = x + ((int64_t)i * h + k) * d;
      int8_t *o = out + ((int64_t)i * h + k) * d;
      for (int j = 0; j < half; j++) {
        int64_t c = cs[i * half + j], s = sn[i * half + j];
        o[j] = sat8(rshift_round(v[j] * c - v[j + half] * s, 14));
        o[j + half] = sat8(rshift_round(v[j] * s + v[j + half] * c, 14));
      }
    }
}

void sb_softmax(const int64_t *scores, const uint8_t *valid,
                const int64_t *lut, int lut_len, int shift, int32_t *out,
                int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    const int64_t *row = scores + (int64_t)i * cols;
    const uint8_t *ok = valid + (int64_t)i * cols;
    int64_t mx = INT64_MIN, total = 0;
    for (int j = 0; j < cols; j++)
      if (ok[j] && row[j] > mx) mx = row[j];
    for (int j = cols - 1; j >= 0; j--) /* any order: integer sum */
      if (ok[j]) {
        int64_t dd = rshift_round(mx - row[j], shift);
        total += lut[dd < lut_len - 1 ? dd : lut_len - 1];
      }
    for (int j = 0; j < cols; j++) {
      int64_t e = 0;
      if (ok[j]) {
        int64_t dd = rshift_round(mx - row[j], shift);
        e = lut[dd < lut_len - 1 ? dd : lut_len - 1];
      }
      int64_t p = (e * 512 + total) / (2 * total);
      out[i * cols + j] = (int32_t)(p < 255 ? p : 255);
    }
  }
}

void sb_add_requant(const int64_t *a, const int64_t *b, int64_t ma,
                    int64_t mb, int s, int8_t *out, int64_t n) {
  for (int64_t i = 0; i < n; i++)
    out[i] = sat8(rshift_round(a[i] * ma + b[i] * mb, s));
}

void sb_mul_requant(const int64_t *a, const int64_t *b, const int64_t *m,
                    const int64_t *s, int8_t *out, int rows, int cols,
                    int per_channel) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) {
      int c = per_channel ? j : 0;
      int64_t idx = (int64_t)i * cols + j;
      out[idx] = sat8(rshift_round(a[idx] * b[idx] * m[c], (int)s[c]));
    }
}

void sb_lut(const int64_t *x, const int8_t *table, int8_t *out, int64_t n) {
  for (int64_t i = 0; i < n; i++) out[i] = table[x[i] + 128];
}

void sb_argmax(const int64_t *x, int64_t *out, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    int best = 0;
    for (int j = 1; j < cols; j++)
      if (x[(int64_t)i * cols + j] > x[(int64_t)i * cols + best]) best = j;
    out[i] = best;
  }
}
