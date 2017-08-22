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

/**
 * note:
 *   'neu' is abbreviation of 'neuron'，'syn' is abbreviation of 'synapse'.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#ifdef WIN32

#include <time.h>

/* winpthread 下未定义 posix_memalign 函数 */
int posix_memalign(void **memptr, size_t alignment, size_t size) {
  *memptr = _aligned_malloc(size, alignment);
  if (*memptr == NULL) return -1;
  return 0;
}

#endif

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

// Maximum 30 * 0.7 = 21M words in the vocabulary
const int vocab_hash_size = 30000000;

// Precision of float numbers
typedef float real;

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2;
int window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = (int) 1e8;
int *table;

void InitUnigramTable() {
  int a;
  long long i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *) malloc(table_size * sizeof(int));
  if (table == NULL) {
    fprintf(stderr, "cannot allocate memory for the table\n");
    exit(1);
  }
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = (real) (pow(vocab[i].cn, power) / (real) train_words_pow);
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real) table_size > d1) {
      i++;
      // 计算下一个词的概率长度
      d1 += pow(vocab[i].cn, power) / (real) train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;  // \r
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *) "</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)
        realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

void DestroyVocab() {
  int a;
  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].word != NULL) {
      free(vocab[a].word);
    }
    if (vocab[a].code != NULL) {
      free(vocab[a].code);
    }
    if (vocab[a].point != NULL) {
      free(vocab[a].point);
    }
  }
  free(vocab[vocab_size].word);
  free(vocab);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a;
  unsigned int hash;
  // Sort the vocabulary order by desc and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  train_words = 0;
  for (a = vocab_size - 1; a > 0; a--) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[a].word);
      vocab[a].word = NULL;
    } else {
      break;
    }
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 1; a < vocab_size; a++) { // Skip </s>
    // Hash will be re-computed, as after the sorting it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
    train_words += vocab[a].cn;
  }
  vocab = (struct vocab_word *)
      realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *) calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *) calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else {
      free(vocab[a].word);
    }
  }
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count =
      (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary =
      (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node =
      (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;  // root node
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];  // reverse
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *) "</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else {
      vocab[i].cn++;
    }
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %zu\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++)
    fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);  // the var c will be \n
    i++;
  }
  fclose(fin);
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %zu\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;

  // syn0 对应投影层连接参数，即词向量
  // syn0_size is vocab_size * layer1_size
  a = posix_memalign((void **) &syn0,
                     128,
                     (long long) vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }

  if (hs) {
    // syn1 对应输出层连接参数
    a = posix_memalign((void **) &syn1,
                       128,
                       (long long) vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (b = 0; b < layer1_size; b++)
      for (a = 0; a < vocab_size; a++)
        syn1[a * layer1_size + b] = 0;
  }

  if (negative > 0) {
    // syn1neg is syn1 of negative sampling. don't name syn1 because we can
    // use both hierarchical softmax and negative sampling in some scenes.
    // syn1neg_size is layer1_size * vocab_size
    a = posix_memalign((void **) &syn1neg,
                       128,
                       (long long) vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (b = 0; b < layer1_size; b++)
      for (a = 0; a < vocab_size; a++)
        syn1neg[a * layer1_size + b] = 0;
  }

  // 随机初始化
  for (b = 0; b < layer1_size; b++)
    for (a = 0; a < vocab_size; a++)
      syn0[a * layer1_size + b] =
          (real) (rand() / RAND_MAX - 0.5) / layer1_size;

  // 构建 huffman tree
  CreateBinaryTree();
}

void DestroyNet() {
#ifdef WIN32
  if (syn0 != NULL) {
    _aligned_free(syn0);
  }
  if (syn1 != NULL) {
    _aligned_free(syn1);
  }
  if (syn1neg != NULL) {
    _aligned_free(syn1neg);
  }
#else
  if (syn0 != NULL) {
    free(syn0);
  }
  if (syn1 != NULL) {
    free(syn1);
  }
  if (syn1neg != NULL) {
    free(syn1neg);
  }
#endif
}

inline real sigmoid(real f) {
  return expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
}

void *TrainModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0,
      sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label;
  unsigned long long next_random = (long long) id;
  real f, g;
  clock_t now;

  // neu1 对应投影层输出，neu1e 对应投影层误差
  real *neu1 = (real *) calloc(layer1_size, sizeof(real));
  real *neu1e = (real *) calloc(layer1_size, sizeof(real));

  // 打开训练语料
  FILE *fi = fopen(train_file, "rb");
  if (fi == NULL) {
    fprintf(stderr, "no such file or directory: %s", train_file);
    exit(1);
  }

  // 按线程切分语料
  fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);

  while (1) {
    // 调整 alpha
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ",
               13,
               alpha,
               word_count_actual / (real) (train_words + 1) * 100,
               word_count_actual
                   / ((real) (now - start + 1) / (real) CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha =
          starting_alpha * (1 - word_count_actual / (real) (train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }

    // 读取语料（一个句子）
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break; // 换行
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1)
              * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long) 25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real) 65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    // 到达训练语料末尾
    if (feof(fi)) break;
    if (word_count > train_words / num_threads) break; // 线程分数据

    // 遍历句子中的词
    word = sen[sentence_position];
    if (word == -1) continue;

    // 初始化
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long) 25214903917 + 11;
    b = next_random % window; // 随机半窗口大小

    if (cbow) {  //train the cbow architecture
      // in -> hidden 正向传播

      //           pos-win+b <- content -> pos+win-b
      //  --+----------+-----------o-----------+----------+--
      // pos-win                  pos                  pos+win
      for (a = b; a < window * 2 + 1 - b; a++) {
        if (a != window) { // 上下文不包含中心词
          c = sentence_position - window + a;

          // 不在句子范围
          if (c < 0) continue;
          if (c >= sentence_length) continue;

          last_word = sen[c];
          if (last_word == -1) continue; // 不在词表中

          l1 = last_word * layer1_size;

          // SUM of syn0(word vector)
          for (c = 0; c < layer1_size; c++)
            neu1[c] += syn0[c + l1];
        }
      }

      // sigmoid and its derivative:
      //   sigmoid(x) = 1 / (1 + e^-x) = e^x / (e^x + 1)
      //   sigmoid'(x) = sigmoid(x) * [1 - sigmoid(x)]
      //
      // derivative of log:
      //   log-a'(x) = 1 / x*ln(a)
      //   ln'(x) = 1 / x
      //
      // in follow, the 'log' is alias of 'ln':
      //   [log(sigmoid(x))]' = 1 / sigmoid(x) * sigmoid'(x)
      //                      = 1 - sigmoid(x)
      //   [log(1 - sigmoid(x))]' = -sigmoid(x)
      //

      // HIERARCHICAL SOFTMAX
      if (hs) {
        // 遍历从根到叶子结点的路径
        for (d = 0; d < vocab[word].codelen; d++) {
          l2 = vocab[word].point[d] * layer1_size;

          // define:
          //   w is center word
          //   C is sentence
          //
          //   x(w) is neuron of projection layer
          //
          //   p(w) is the path from root to leaf which represent 'w'
          //   l(w) is the length of p(w)
          //   p(w)_1, p(w)_2,...,p(w)_l(w) is the nodes of p(w), p(w)_1 is root
          //   d(w)_2, d(w)_3,...,d(w)_l(w) is the huffman code of p(w)
          //   theta(w)_1, theta(w)_2,..., theta(w)_l(w)-1 is arguments of
          //     non-leaf nodes in p(w)
          //
          //   p(w | Context(w)) = TT j in [2...l(w)]: p(d-w_j | x-w, theta-w_j-1)
          //
          //                                 +- sigmoid(x-w * theta-w_j-1),     if d(w)_j = 0
          //   p(d-w_j | x-w, theta-w_j-1) = |
          //                                 +- 1 - sigmoid(x-w * theta-w_j-1), if d(w)_j = 1
          //
          //   p(d-w_j | x-w, theta-w_j-1) =
          //       sigmoid(x-w * theta-w_j-1) ^ [1 - d-w_j]
          //           * [1 - sigmoid(x-w * theta-w_j-1)] ^ d-w_j
          //
          // so,
          //   loss = Sigma w in C: log p(w | Context(w))
          //        = Sigma w in C: log TT j in [2...l(w)]: p(w | Context(w))
          //        = Sigma w in C: log TT j in [2...l(w)]:
          //            {sigmoid(x-w * theta-w_j-1) ^ [1 - d-w_j]
          //                * [1 - sigmoid(x-w * theta-w_j-1)] ^ d-w_j}
          //        = Sigma w in C: Sigma j in [2...l(w)]:
          //            {(1 - d-w_j) * log[sigmoid(x-w * theta-w_j-1)]
          //                + d-w_j * log[1 - sigmoid(x-w * theta-w_j-1)]}
          //
          // define part of loss and gradient in here is:
          //   L1(w, j) = (1 - d-w_j) * log[sigmoid(x-w * theta-w_j-1)]
          //                + d-w_j * log[1 - sigmoid(x-w * theta-w_j-1)]
          //
          //   gradient(theta-w_j-1) = derivative(L1, theta-w_j-1)
          //       = (1 - d-w_j) * [1 - sigmoid(x-w * theta-w_j-1)] * x-w
          //           - d-w_j * sigmoid(x-w * theta-w_j-1) * x-w
          //       = {(1 - d-w_j) * [1 - sigmoid(x-w * theta-w_j-1)]
          //           - d-w_j * sigmoid(x-w * theta-w_j-1)} * x-w
          //       = [1 - d-w_j - sigmoid(x-w * theta-w_j-1)] * x-w
          //
          //   theta-w_j-1 = theta-w_j-1 + eta * gradient(theta-w_j-1)
          //       = theta-w_j-1
          //           + eta * [1 - d-w_j - sigmoid(x-w * theta-w_j-1)] * x-w
          //
          //   gradient(x-w) = derivative(L1, x-w)
          //       = [1 - d-w_j - sigmoid(x-w * theta-w_j-1)] * theta-w_j-1
          //
          // in the following text:
          //   alpha is eta
          //   f is sigmoid(x-w * theta-u)
          //   g is [1 - d-w_j - sigmoid(x-w * theta-w_j-1)] * eta
          //
          //   neu1 is x(w)
          //   neu1e is error of neu1
          //   syn1 is theta(w)
          //

          // Propagate hidden -> output
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP || f >= MAX_EXP) continue;
          f = sigmoid(f);

          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;

          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
      }

      // NEGATIVE SAMPLING
      if (negative > 0) {
        // 采样 negative 个负样本 + 1 个正样本
        for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            // 正样本
            target = word;
            label = 1;
          } else {
            // 负样本
            next_random = next_random * (unsigned long long) 25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }

          // target is one-hot code of input layer
          l2 = target * layer1_size;

          // define:
          //   w is center word
          //   C is sentence
          //   NEG(w) is negative samples set of 'w'
          //
          //   x(w) is neuron of projection layer
          //   theta(w) is synapse between projection layer and output layer
          //
          //   g(w) = p(Context(w) | w)
          //        = TT u in {w} + NEG(w): p(u | Context(w))
          //
          //            +- 1,  if u == w
          //   L-w(u) = |
          //            +- 0,  if u != w
          //
          //                       +- sigmoid(x-w * theta-u),     if L-w(u) == 1
          //   p(u | Context(w)) = |
          //                       +- 1 - sigmoid(x-w * theta-u), if L-w(u) == 0
          //
          //   p(u | Context(w)) = sigmoid(x-w * theta-u) ^ L-w(u)
          //       * [1 - sigmoid(x-w * theta-u)] ^ [1 - L-w(u)]
          //
          // so,
          //   g(w) = p(w | Context(w)) * TT u in NEG(w): p(u | Context(w))
          //        = sigmoid(x-w * theta-w)
          //            * TT u in NEG(w): [1 - sigmoid(x-w * theta-u)]
          //
          //   G = TT w in C: g(w)
          //
          //   loss = log G
          //        = log TT w in C: g(w)
          //        = Sigma w in C: log g(w)
          //        = Sigma w in C: log p(Context(w) | w)
          //        = Sigma w in C: log TT u in {w} + NEG(w): p(u | Context(w))
          //        = Sigma w in C: Sigma u in {w} + NEG(w):
          //            L-w(u) * log[sigmoid(x-w * theta-u)]
          //                + [1 - L-w(u)] * log[1 - sigmoid(x-w * theta-u)]
          //
          // define part of loss and gradient in here is:
          //   L1(w, u) = L-w(u) * log[sigmoid(x-w * theta-u)]
          //                + (1 - L-w(u)) * log[1 - sigmoid(x-w * theta-u)]
          //
          //   gradient(theta-u) = derivative(L1, theta-u)
          //       = L-w(u) * [1 - sigmoid(x-w * theta-u)] * x-w
          //           - [1 - L-w(u)] * sigmoid(x-w * theta-u) * x-w
          //       = {L-w(u) * [1 - sigmoid(x-w * theta-u)]
          //           - [1 - L-w(u)] * sigmoid(x-w * theta-u)} * x-w
          //       = [L-w(u) - sigmoid(x-w * theta-u)] * x-w
          //
          //   theta-u = theta-u + eta * gradient(theta-u)
          //           = theta-u + eta * [L-w(u) - sigmoid(x-w * theta-u)] * x-w
          //
          //   gradient(x-u) = derivative(L1, x-w)
          //       = [L-w(u) - sigmoid(x-w * theta-u)] * theta-u
          //
          // in the following text:
          //   alpha is eta
          //   f is x-w * theta-u
          //   g is [L-w(u) - sigmoid(x-w * theta-u)] * eta
          //
          //   neu1 is x(w)
          //   neu1e is error of neu1
          //   syn1neg is theta(u)
          //

          // 线性和，没偏置？
          f = 0;
          for (c = 0; c < layer1_size; c++)
            f += neu1[c] * syn1neg[c + l2];

          // 计算公共项 g
          if (f > MAX_EXP) {
            g = (label - 1) * alpha;
          } else if (f < -MAX_EXP) {
            g = (label - 0) * alpha;
          } else {
            g = (label - sigmoid(f)) * alpha;
          }

          // 更新参数，注意更新顺序
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
      }

      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; a++) {
        if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++)
            syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } else {  //train skip-gram
      // 遍历上下文
      for (a = b; a < window * 2 + 1 - b; a++) {
        if (a != window) { // 上下文不包含中心词
          c = sentence_position - window + a;

          // 不在句子范围
          if (c < 0) continue;
          if (c >= sentence_length) continue;

          last_word = sen[c];
          if (last_word == -1) continue; // 不在词表中

          l1 = last_word * layer1_size;

          for (c = 0; c < layer1_size; c++)
            neu1[c] = syn0[c + l1];

          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

          // HIERARCHICAL SOFTMAX
          if (hs) {
            for (d = 0; d < vocab[word].codelen; d++) {
              l2 = vocab[word].point[d] * layer1_size;

              // Propagate hidden -> output
              f = 0;
              for (c = 0; c < layer1_size; c++)
                f += neu1[c] * syn1[c + l2];
              if (f <= -MAX_EXP || f >= MAX_EXP) continue;
              f = sigmoid(f);

              // 'g' is the gradient multiplied by the learning rate
              g = (1 - vocab[word].code[d] - f) * alpha;

              // Propagate errors output -> hidden
              for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
              // Learn weights hidden -> output
              for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
            }
          }

          // NEGATIVE SAMPLING
          if (negative > 0) {
            for (d = 0; d < negative + 1; d++) {
              if (d == 0) {
                target = word;
                label = 1;
              } else {
                next_random =
                    next_random * (unsigned long long) 25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == 0) target = next_random % (vocab_size - 1) + 1;
                if (target == word) continue;
                label = 0;
              }
              l2 = target * layer1_size;

              f = 0;
              for (c = 0; c < layer1_size; c++)
                f += neu1[c] * syn1neg[c + l2];

              if (f > MAX_EXP) {
                g = (label - 1) * alpha;
              } else if (f < -MAX_EXP) {
                g = (label - 0) * alpha;
              } else {
                g = (label - sigmoid(f)) * alpha;
              }

              for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
              for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
            }
          }

          // Learn weights input -> hidden
          for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        }
      }
    }

    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
  return 0;
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;

  // 创建线程
  pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
  if (pt == NULL) {
    fprintf(stderr, "cannot allocate memory for threads\n");
    exit(1);
  }

  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;

  // 构造词表
  if (read_vocab_file[0] != 0)
    ReadVocab();
  else
    LearnVocabFromTrainFile();

  // 保存词表
  if (save_vocab_file[0] != 0) SaveVocab();

  if (output_file[0] == 0) return;

  // 初始化网络结构
  InitNet();

  // 初始化线性概率采样表
  if (negative > 0) InitUnigramTable();

  // 启动线程，训练网络
  start = clock();
  for (a = 0; a < num_threads; a++)
    pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
  for (a = 0; a < num_threads; a++)
    pthread_join(pt[a], NULL);

  fo = fopen(output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
    exit(1);
  }

  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      if (vocab[a].word != NULL) {
        fprintf(fo, "%s ", vocab[a].word);
      }
      if (binary) {
        for (b = 0; b < layer1_size; b++)
          fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      } else {
        for (b = 0; b < layer1_size; b++)
          fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      }
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *) malloc(classes * sizeof(int));
    if (centcn == NULL) {
      fprintf(stderr, "cannot allocate memory for centcn\n");
      exit(1);
    }
    int *cl = (int *) calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *) calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) {
          cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
          centcn[cl[c]]++;
        }
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++)
            x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++)
      fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }

  fclose(fo);
  free(table);
  free(pt);

  // 销毁词表
  DestroyVocab();
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) {
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;

  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous back of words model; default is 0 (skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }

  // get arguments
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *) "-size", argc, argv)) > 0)
    layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-train", argc, argv)) > 0)
    strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0)
    strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0)
    strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-debug", argc, argv)) > 0)
    debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-binary", argc, argv)) > 0)
    binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-cbow", argc, argv)) > 0)
    cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0)
    alpha = (real) atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-output", argc, argv)) > 0)
    strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-window", argc, argv)) > 0)
    window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-sample", argc, argv)) > 0)
    sample = (real) atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-hs", argc, argv)) > 0)
    hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-negative", argc, argv)) > 0)
    negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-threads", argc, argv)) > 0)
    num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0)
    min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-classes", argc, argv)) > 0)
    classes = atoi(argv[i + 1]);

  printf("train file: %s\n", train_file);
  printf("output file: %s\n", output_file);

  // alloc memory
  vocab = (struct vocab_word *)
      calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
  expTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (expTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }

  // NOTE: MAX_EXP is 6, EXP_TABLE_SIZE is 1000
  //
  // region is [-MAX_EXP, MAX_EXP], map to [0, EXP_TABLE_SIZE]
  // so, step is 2*MAX_EXP / EXP_TABLE_SIZE
  //
  // region to index is:
  //   i = (r + MAX_EXP) * (EXP_TABLE_SIZE / 2*MAX_EXP)
  //     = (r + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)
  //
  // index to region is:
  //   r = i / (EXP_TABLE_SIZE / 2*MAX_EXP) - MAX_EXP
  //     = (i * 2*MAX_EXP) / EXP_TABLE_SIZE - MAX_EXP
  //     = (i * 2 / EXP_TABLE_SIZE - 1) * MAX_EXP
  //     = (i / EXP_TABLE_SIZE * 2 - 1) * MAX_EXP
  //
  // approximation of exponential function:
  //   e^z = 1 + z/1! + z^2/2! + z^3/3! + ...
  // but we use library function 'exp'.
  //
  // sigmoid function:
  //   sigmoid(x) = 1 / (1 + e^-x)
  //              = e^x / (e^x + 1)
  //
  for (i = 0; i <= EXP_TABLE_SIZE; i++) {
    // pre-compute the exp() table
    expTable[i] = (real) exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
    // pre-compute f(x) = x / (x + 1)
    expTable[i] = expTable[i] / (expTable[i] + 1);
  }
  // f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

  TrainModel();
  DestroyNet();

  // free memory
  free(vocab_hash);
  free(expTable);

  return 0;
}
