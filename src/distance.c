//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h> // mac os x


const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char bestw[N][max_size];
  char file_name[max_size], st[100][max_size];
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, c, d, cn, bi[100];
  char ch;
  float *M;  // 单位向量
  char *vocab;  // vocabulary

  if (argc < 2) {
    printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }

  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }

  // 读取 word projections
  fscanf(f, "%lld", &words);  // 读取词表数量
  fscanf(f, "%lld", &size);  // 读取词向量长度
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));  // 给词表分配内存
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));  // 给向量表分配内存
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
           (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = (float) sqrt(len);  // 欧氏长度
    for (a = 0; a < size; a++) M[a + b * size] /= len;  // 单位向量
  }
  fclose(f);

  // 主循环
  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    printf("Enter word or sentence (EXIT to break): ");
    a = 0;
    while (1) {
      st1[a] = (char) fgetc(stdin);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }

    // 退出
    if (!strcmp(st1, "EXIT")) break;

    // 以空格切分输入
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;

    // 定位输入在词表中的索引
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == -1) continue;

    printf("\n                                              Word       Cosine distance");
    printf("\n------------------------------------------------------------------------\n");

    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }

    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = (float) sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;  // 单位向量

    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;

    for (c = 0; c < words; c++) {
      a = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
      if (a == 1) continue;  // 输入词，跳过

      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];  // 余弦

      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          // 插入排序
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &vocab[c * max_w]);
          break;
        }
      }
    }

    for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
  }
  return 0;
}
