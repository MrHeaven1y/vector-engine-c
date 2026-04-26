# C Autograd Engine — API Reference

A scalar-batched reverse-mode autograd engine written in plain C.  
Every tensor is a flat `double[]` column-vector; a "batch" is those scalars laid side-by-side in one array.

---

## What’s implemented

- Dynamic computation graph (GraphContext)
- Reverse-mode autodiff
- Tensor operations with broadcasting
- Neural network layers (Dense, Linear)
- Optimizer (SGD + momentum)

## Project Layout

```
project/
├── include/          ← headers  (graph.h, ops.h, layers.h, optimizers.h, stensor.h)
├── src/              ← library  (graph.c, ops.c, layers.c, optimizers.c, tensor.c, train.c, io.c)
├── main.c            ← YOUR code lives here, at the project root
└── README.md
```

**You never touch `src/` or `include/` during normal use.**  
Write your entry point in `main.c` (or any `.c` at root level) and compile with:

```bash
gcc main.c src/*.c -Iinclude -lm -o net
```

---

## Challenges

- Manual memory management in C
- Graph lifecycle handling (reset_tape vs reset_graph)
- Gradient accumulation across batches

## Core Concepts

| Concept | What it is |
|---|---|
| `Tensor` | A node in the compute graph.  Holds `data[]`, `grad[]`, and pointers to child nodes. |
| `GraphContext` | Owns three registries: `tape` (op nodes), `params` (learnable weights), `inputs` (data nodes). |
| `Model` | A flat list of `Tensor*` params registered by layers — handed to the optimizer. |
| `Optimizer` | Holds param pointers, velocity buffers, and a `step` function pointer. |

---

## 1. Graph Context

```c
#include "graph.h"
```

### `init_graph`

```c
GraphContext* init_graph(int t_cap, int p_cap, int i_cap);
```

Allocates a new context.  All three registries auto-resize (double capacity) when full, so initial caps are just hints.

| Argument | Meaning |
|---|---|
| `t_cap` | Initial capacity for op nodes on the tape |
| `p_cap` | Initial capacity for learnable parameters |
| `i_cap` | Initial capacity for input nodes |

**Rule of thumb:** `t_cap = layers × neurons × batch × 4`, `p_cap = total_weights`, `i_cap = input_dim + output_dim`.

---

### Tensor Constructors

```c
Tensor* Input(GraphContext* ctx, double* values, int size);
```
Creates a data node (not differentiated through, but grad array is allocated so downstream nodes can push into it).

```c
Tensor* Param(GraphContext* ctx, double* values, int size);
```
Creates a learnable parameter from existing values.

```c
Tensor* RandParam(GraphContext* ctx, int size);
```
Creates a learnable parameter initialised with Xavier uniform:  
`data[i] ∈ [-√(6/size), +√(6/size)]`.

---

### Resetting the Graph

```c
void reset_graph(GraphContext* ctx);     /* frees tape + input nodes; keeps params */
void reset_tape(GraphContext* ctx);      /* frees op nodes only                    */
void reset_inputs(GraphContext* ctx);    /* frees input nodes only                 */
void reset_computation(GraphContext* ctx); /* frees params + inputs                */
```

**Inside the training loop** call `reset_tape(ctx)` — frees only op nodes while keeping input tensors and params alive across epochs.  
Call `reset_graph(ctx)` when fully done (e.g. after inference) to also free input nodes.

---

## 2. Operations

```c
#include "ops.h"
```

Every op takes a `GraphContext*` as first argument, allocates a new node, wires the backward function, and returns the result tensor.  All ops support **scalar broadcast**: if one operand has `size == 1` it is broadcast across the other.

### Arithmetic

```c
Tensor* Add(GraphContext* ctx, Tensor* a, Tensor* b);
Tensor* Sub(GraphContext* ctx, Tensor* a, Tensor* b);
Tensor* Mul(GraphContext* ctx, Tensor* a, Tensor* b);
Tensor* Div(GraphContext* ctx, Tensor* a, Tensor* b);
Tensor* Neg(GraphContext* ctx, Tensor* a);           /* element-wise negate */
Tensor* Square(GraphContext* ctx, Tensor* a);
```

### Math

```c
Tensor* Exp(GraphContext* ctx, Tensor* a);
Tensor* Log(GraphContext* ctx, Tensor* a);           /* natural log */
Tensor* Mean(GraphContext* ctx, Tensor* a);          /* → size-1 tensor */
```

### Activations

```c
Tensor* Relu(GraphContext* ctx, Tensor* a);
Tensor* Tanh(GraphContext* ctx, Tensor* a);
Tensor* Sigmoid(GraphContext* ctx, Tensor* a);
Tensor* Softmax(GraphContext* ctx, Tensor* a, int sample_width);
```

For `Softmax`, `a` is a flat array of all logits for all samples, and `sample_width` is the number of classes per sample.

### Loss Functions

```c
Tensor* Mse(GraphContext* ctx, Tensor* y, Tensor* y_hat);       /* mean squared error  */
Tensor* Rmse(GraphContext* ctx, Tensor* y, Tensor* y_hat);      /* root MSE            */
Tensor* Mae(GraphContext* ctx, Tensor* y, Tensor* y_hat);       /* mean absolute error */
Tensor* Binary_CE(GraphContext* ctx, Tensor* target, Tensor* pred); /* binary cross-entropy */
Tensor* Softmax_CE(GraphContext* ctx, Tensor* target, Tensor* pred);/* fused softmax + CE   */
```

### Graph-level Helpers (in `graph.c`)

```c
/* Build a softmax over an array of per-class logit tensors */
Tensor** build_softmax(GraphContext* ctx, Tensor** logits, int num_classes);

/* Build categorical cross-entropy from probability and target arrays */
Tensor* build_categorical_CE(GraphContext* ctx,
                              Tensor** probs, Tensor** targets,
                              int num_classes);
```

These are the recommended path for multi-class classification.  `build_softmax` returns a heap-allocated `Tensor**` — call `free()` on the pointer array after use (the tensors themselves live on the tape).

---

## 3. Backward Pass

```c
#include "stensor.h"   /* Tensor type */
/* defined in tensor.c, declared in graph.h */

void backward(GraphContext* ctx, Tensor* root);
```

Runs topological sort from `root`, then iterates in reverse calling each node's registered `op_backward`.

**Before calling `backward`:**  
Seed the output gradient manually if you want something other than all-ones:

```c
for (int i = 0; i < loss->size; i++) loss->grad[i] = -1.0 / BATCH;
backward(ctx, loss);
```

If `root->grad` is all-zeros when `backward` is called, it is auto-filled with `1.0`.

---

## 4. Layers

```c
#include "layers.h"
```

### Model

```c
Model* init_model(int capacity);
void   register_param(Model* m, Tensor* param);
```

`Model` is just a flat array of `Tensor*` params. Pass `m->params` and `m->count` to the optimizer.

### LinearLayer — single weight + bias, element-wise

```c
LinearLayer* _linear_layer(GraphContext* ctx, Model* m, int size);
Tensor*      Linear(GraphContext* ctx, LinearLayer* layer, Tensor* x);
```

Computes `w * x + b` element-wise.  Weight and bias are `size`-length vectors (one per batch lane).

### DenseLayer — fully-connected

```c
DenseLayer* _dense_layer(GraphContext* ctx, Model* m,
                          int input_dim, int output_dim, int batch_size);

Tensor** Dense(GraphContext* ctx, DenseLayer* dlayer, Tensor** inputs);
```

`inputs` is an array of `input_dim` tensors each of length `batch_size`.  
Returns an array of `output_dim` tensors each of length `batch_size`.  
The caller is responsible for `free()`-ing the returned pointer array.

### Resizing for Inference

```c
void resize_model_batch(Model* m, int new_batch_size);
```

Call this after training to collapse weight tensors from `[batch_size]` down to `[1]` (or any new size).  The first value (`data[0]`) — the learned scalar — is broadcast to fill the new size.

---

## 5. Optimizer

```c
#include "optimizers.h"
```

### Create SGD w/ Momentum

```c
Optimizer* sgd(Tensor** params, int count,
               int batch_size, double lr, double beta);
```

| Argument | Meaning |
|---|---|
| `params` | `m->params` array |
| `count` | `m->count` |
| `batch_size` | batch size used during training |
| `lr` | learning rate |
| `beta` | momentum coefficient (0 = plain SGD, 0.9 = typical) |

### Step

```c
opt->step(opt);   /* call after backward + sync_weights */
```

Internally calls `_sgd_momentum`, which also zeroes gradients after the update — no need to call `zero_grad` separately when using momentum SGD.

### Utilities

```c
void sync_weights(Optimizer* optim);             /* sum gradients across batch lanes → lane 0 */
void clip_grad_norm(Optimizer* optim, double max_norm); /* global gradient norm clipping */
void zero_grad(Optimizer* optim);                /* zero all param gradients */
```

**Canonical per-epoch order:**

```c
backward(ctx, loss);
opt->step(opt);     /* _sgd_momentum calls sync_weights internally */
reset_tape(ctx);    /* frees op nodes; inputs + params stay alive  */
free(layer_out);    /* free the Tensor** pointer arrays from Dense  */
```

`clip_grad_norm` and `zero_grad` are available as standalone utilities but are not required when using `sgd` — the momentum step zeroes gradients itself after updating.

---

## 6. Utilities

```c
void print_tensor(Tensor* v);   /* prints data column */
void shape(Tensor* v);          /* prints (size, 1)   */
```

---

## 7. Full Example — XOR (`xor_train.c`)

This is the reference example. It lives at the **project root** alongside `README.md`.

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "graph.h"
#include "ops.h"
#include "layers.h"
#include "optimizers.h"
#include "stensor.h"
#include "io.h"

int main(){
    srand(time(NULL));

    printf("--- Vector Engine: XOR Training ---\n\n");

    GraphContext* ctx = init_graph(10000, 1000, 100);
    Model* model = init_model(100);

    int batch_size = 4;

    /* full dataset fits in one batch — inputs created ONCE, reused every epoch */
    double x0_data[] = {0.0, 0.0, 1.0, 1.0};
    double x1_data[] = {0.0, 1.0, 0.0, 1.0};
    double y_data[]  = {0.0, 1.0, 1.0, 0.0};

    Tensor* in_tensors[2];
    in_tensors[0] = Input(ctx, x0_data, batch_size);
    in_tensors[1] = Input(ctx, x1_data, batch_size);
    Tensor* target = Input(ctx, y_data, batch_size);

    /* layers registered into model automatically via _dense_layer */
    DenseLayer* d1 = _dense_layer(ctx, model, 2, 4, batch_size);
    DenseLayer* d2 = _dense_layer(ctx, model, 4, 1, batch_size);

    Optimizer* opt = sgd(model->params, model->count, batch_size, 0.5, 0.9);

    int epochs = 6000;
    for(int epoch = 0; epoch <= epochs; epoch++){

        Tensor** h1 = Dense(ctx, d1, in_tensors);

        Tensor* h1_act[4];
        for(int i = 0; i < 4; i++) h1_act[i] = Tanh(ctx, h1[i]);

        Tensor** out = Dense(ctx, d2, h1_act);
        Tensor* pred = Sigmoid(ctx, out[0]);
        Tensor* loss = Mse(ctx, target, pred);

        backward(ctx, loss);   /* grad seeded automatically (all 1.0) */
        opt->step(opt);        /* sync_weights + momentum update + zero_grad */

        if(epoch % 100 == 0)
            printf("Epoch %d | MSE Loss: %f\n", epoch, loss->data[0]);

        /* free op nodes only — inputs and params survive to next epoch */
        reset_tape(ctx);
        free(h1);
        free(out);
    }

    /* collapse weight tensors from [batch_size] → [1] before inference */
    resize_model_batch(model, 1);
    printf("\n--- Inference (batch_size = 1) ---\n");

    double test_x0[] = {1.0};
    double test_x1[] = {0.0};

    Tensor* single_input[2];
    single_input[0] = Input(ctx, test_x0, 1);
    single_input[1] = Input(ctx, test_x1, 1);

    Tensor** h1_eval   = Dense(ctx, d1, single_input);
    Tensor*  h1_act_eval[4];
    for(int i=0; i<4; i++) h1_act_eval[i] = Tanh(ctx, h1_eval[i]);

    Tensor** out_eval  = Dense(ctx, d2, h1_act_eval);
    Tensor*  pred_eval = Sigmoid(ctx, out_eval[0]);

    printf("Input: (1.0, 0.0) | Predicted: %.4f\n", pred_eval->data[0]);

    /* save graph rooted at pred_eval; 1000 = max node budget */
    save_model(pred_eval, 1000, "xor_model.json");

    free(h1_eval);
    free(out_eval);

    reset_graph(ctx);   /* free remaining tape + input nodes */
    free(ctx);
    free(model->params);
    free(model);

    return 0;
}
```

### What the example shows

- **Inputs outside the loop** — when the full dataset fits in one batch, create `Input` tensors once and reuse them every epoch. `reset_tape` leaves them untouched.
- **`reset_tape` vs `reset_graph`** — use `reset_tape` inside the training loop (keeps inputs alive), `reset_graph` after inference (frees everything except params).
- **`opt->step` is self-contained** — no manual `sync_weights` call needed; `_sgd_momentum` handles it internally.
- **`resize_model_batch` before inference** — mandatory when switching from batch > 1 to single-sample prediction.
- **`save_model`** — pass the root output tensor, a node budget, and a filename; the graph is serialised to JSON.

---

## Common Pitfalls

**Freeing the wrong thing after `Dense` / `build_softmax`**  
Only `free()` the pointer array (`Tensor**`), never the individual `Tensor*` nodes — those are owned by the tape and freed by `reset_tape`/`reset_graph`.

**Using `reset_graph` inside the training loop**  
`reset_graph` frees input nodes too. Use `reset_tape` inside the loop so your pre-built `Input` tensors survive to the next epoch. Use `reset_graph` only after you are fully done.

**Calling `sync_weights` manually before `opt->step`**  
When using `sgd`, `opt->step` calls `_sgd_momentum` which calls `sync_weights` at the top. Calling it again manually will double-accumulate gradients.

**Not resizing before inference**  
Weights are stored as `[batch_size]` arrays during training. Call `resize_model_batch(m, 1)` before running single-sample inference.

**Capacity too small**  
If `t_cap / p_cap / i_cap` are too small, `realloc` kicks in every step. Set them to ~2× your expected count upfront.


<img width="497" height="525" alt="Image" src="https://github.com/user-attachments/assets/e8a582f6-37c4-4378-98cf-120a67c5dcc7" />
