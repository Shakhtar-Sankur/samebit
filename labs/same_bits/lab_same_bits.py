import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="The Same-Bits Contract")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    _here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    import engine

    return engine, mo


@app.cell
def _(mo):
    mo.md(r"""
    # The Same-Bits Contract
    ### Why temperature 0 does not give the same answer twice, and what it costs to make it

    **The question.** You send the same prompt to a language model at temperature 0, so every
    token is the most likely one and nothing is sampled. You send it again. Do you get the same
    answer?

    In September 2025, Thinking Machines Lab sent one prompt 1,000 times to Qwen3-235B at
    temperature 0 and got **80 different completions**. Nothing was random. The server
    batched each request with different neighbours, the kernels picked a different reduction
    order for each batch size, and floating-point addition gave slightly different bits. The
    difference grew until one greedy choice flipped.

    In this lab you reproduce that chain step by step, with real IEEE 754 arithmetic in your
    browser:

    | Part | You discover |
    |---|---|
    | **A** One sum, five orders | floating-point addition is not associative |
    | **B** The batch picks the kernel | a request's outputs depend on who it is batched with |
    | **C** From bits to words | tiny bit differences change generated text |
    | **D** Batch-invariant kernels | the first fix, and its price |
    | **E** Integer arithmetic | the second fix, and its price |

    Each part asks for a prediction first. Lock it, then use the instruments to test it.

    *Prerequisites:* floating-point formats, matrix multiplication, greedy decoding.
    """)
    return


@app.cell
def _(mo):
    a_prediction = mo.ui.radio(
        {"1: addition is addition": "1", "2 or 3": "2-3", "4 or 5": "4-5"},
        label="You add the same 4,096 numbers in five different orders. How many different results?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Identical bits": "same", "A few outputs differ": "few", "Most outputs differ": "most"},
        label="Your request is served alone, then batched with 31 others. Its 128 outputs are:",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"None": "0", "1 to 10": "1-10", "More than 10": "10+"},
        label="64 users at temperature 0, served alone and then 16 at a time. How many get different text?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"None": "0", "Fewer than before": "fewer", "As many as before": "same"},
        label="The kernel now uses the same split for every batch size. How many users get different text?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Identical at every batch size": "same", "Still different": "differ"},
        label="int8 weights and activations, exact int32 sums, any order. Alone versus batched:",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo):
    a_dtype = mo.ui.dropdown(["float16", "float32"], value="float32", label="Format")
    a_seed = mo.ui.slider(0, 20, value=0, label="Data seed")
    b_batch = mo.ui.slider(1, 64, value=32, label="Batch size")
    b_dtype = mo.ui.dropdown(["float16", "float32"], value="float16", label="Format")
    c_batch = mo.ui.slider(2, 64, value=16, step=2, label="Batch size")
    c_dtype = mo.ui.dropdown(["float16", "float32"], value="float16", label="Format")
    d_batch = mo.ui.slider(2, 64, value=16, step=2, label="Batch size")
    e_batch = mo.ui.slider(2, 64, value=16, step=2, label="Batch size")
    return (
        a_dtype,
        a_seed,
        b_batch,
        b_dtype,
        c_batch,
        c_dtype,
        d_batch,
        e_batch,
    )


@app.cell
def _(a_dtype, a_seed, engine):
    a_result = engine.addition_orders(n=4096, dtype=a_dtype.value, seed=a_seed.value)
    return (a_result,)


@app.cell
def _(b_batch, b_dtype, engine):
    b_result = engine.batch_contrast(b_batch.value, dtype=b_dtype.value)
    b_default = engine.batch_contrast(32, dtype="float16")
    return b_default, b_result


@app.cell
def _(c_batch, c_dtype, engine):
    c_result = engine.serving_contrast(c_batch.value, dtype=c_dtype.value)
    c_default = engine.serving_contrast(16, dtype="float16")
    return c_default, c_result


@app.cell
def _(d_batch, engine):
    d_result = engine.serving_contrast(d_batch.value, policy="invariant")
    d_extra = {
        b: (engine.extra_reduction_adds(b, "throughput"), engine.extra_reduction_adds(b, "invariant"))
        for b in (1, 4, 16, 64)
    }
    return d_extra, d_result


@app.cell
def _(e_batch, engine):
    e_result = engine.serving_contrast(e_batch.value, mode="integer")
    e_quality = engine.integer_agreement()
    return e_quality, e_result


@app.cell
def _(mo):
    def locked(form):
        return mo.callout(mo.md(f"**Lock your prediction first.**\n\n{form}"), kind="neutral")

    def verdict(form, truth, text):
        right = form.value == truth
        return mo.callout(
            mo.md(("**Your prediction was right.** " if right else "**Not what you predicted.** ") + text),
            kind="success" if right else "warn",
        )

    return locked, verdict


@app.cell
def _(a_dtype, a_prediction, a_result, a_seed, locked, mo, verdict):
    _truth = "1" if a_result["distinct"] == 1 else ("2-3" if a_result["distinct"] <= 3 else "4-5")
    if a_prediction.value is None:
        part_a = mo.vstack([a_prediction, locked("Five orders: left to right, a pairwise tree, 8 and 64 parallel blocks, and reversed.")])
    else:
        _rows = "\n".join(
            f"| {r['order']} | {r['value']:.9g} | `{r['bits']}` | {r['ulps_from_left_to_right']} |"
            for r in a_result["rows"]
        )
        part_a = mo.vstack([
            a_prediction,
            mo.hstack([a_dtype, a_seed]),
            mo.md(
                f"""
    | Order | Result | Bits (SHA-256) | Steps from left-to-right |
    |---|---|---|---|
    {_rows}

    The exact sum, in float64, is **{a_result['exact']:.12g}**. The five orders give
    **{a_result['distinct']} different {a_dtype.value} results**.
    """
            ),
            verdict(
                a_prediction, _truth,
                "Each addition rounds to the nearest representable number, so (a + b) + c and a + (b + c) "
                "can round differently. A GPU never adds 4,096 numbers left to right: it splits them across "
                "threads and blocks, and the split decides the order. None of these results is wrong; they "
                "differ in the last bits, and no order is the correct one.",
            ),
        ])
    return (part_a,)


@app.cell
def _(
    b_batch,
    b_default,
    b_dtype,
    b_prediction,
    b_result,
    locked,
    mo,
    verdict,
):
    _n = b_default["outputs_differing"]
    _truth = "same" if _n == 0 else ("few" if _n < 64 else "most")
    if b_prediction.value is None:
        part_b = mo.vstack([b_prediction, locked(
            "The kernel computes one output row of a 256 × 128 matrix multiplication. Like tuned GPU "
            "libraries, it splits the 256-long reduction into 4 pieces when the batch is too small to fill "
            "the chip, 2 pieces at 4 to 15 rows, and none from 16 rows up."
        )])
    else:
        part_b = mo.vstack([
            b_prediction,
            mo.hstack([b_batch, b_dtype]),
            mo.md(
                f"""
    | | Served alone | Served in a batch of {b_batch.value} |
    |---|---|---|
    | reduction split into | {b_result['splits_alone']} | {b_result['splits_batched']} |
    | bits of the 128 outputs | `{b_result['digest_alone']}` | `{b_result['digest_batched']}` |

    **{b_result['outputs_differing']} of {b_result['outputs']} outputs differ**, by up to
    {b_result['max_ulps']} representable steps ({b_result['max_abs']:.2e} in value).
    """
            ),
            verdict(
                b_prediction, _truth,
                f"At batch 32 in float16, {_n} of 128 outputs change. Your row's numbers are the same in both "
                "runs; only the batch around it changed, and with it the kernel's reduction order. This is "
                "batch variance: in a server, your result depends on other people's requests.",
            ),
        ])
    return (part_b,)


@app.cell
def _(
    c_batch,
    c_default,
    c_dtype,
    c_prediction,
    c_result,
    locked,
    mo,
    verdict,
):
    _n = c_default["changed"]
    _truth = "0" if _n == 0 else ("1-10" if _n <= 10 else "10+")
    if c_prediction.value is None:
        part_c = mo.vstack([c_prediction, locked(
            "A small language model (64-wide, 76-token vocabulary, seeded weights) decodes 40 tokens greedily "
            "for 64 users. Its output layer uses the kernel from Part B."
        )])
    else:
        _ex = "\n".join(
            f"| {x['user']} | {x['first_divergent_token']} | {x['alone']} | {x['batched']} |"
            for x in c_result["examples"]
        ) or "| | | | |"
        part_c = mo.vstack([
            c_prediction,
            mo.hstack([c_batch, c_dtype]),
            mo.md(
                f"""
    **{c_result['changed']} of {c_result['users']} users** get different text at batch {c_batch.value}
    in {c_dtype.value}.

    | User | First different token | Served alone | Served in a batch |
    |---|---|---|---|
    {_ex}
    """
            ),
            verdict(
                c_prediction, _truth,
                f"At batch 16 in float16, {_n} of 64 users get different text. A difference in the last bits "
                "only matters when two tokens are nearly tied, but over many tokens and users some are. Once "
                "one token flips, the rest of the text follows a different path. Switch to float32: at this "
                "toy scale the flips disappear. At real scale they do not: Thinking Machines saw 80 "
                "different completions in 1,000 runs of a 235B model.",
            ),
        ])
    return (part_c,)


@app.cell
def _(d_batch, d_extra, d_prediction, d_result, locked, mo, verdict):
    _truth = "0" if d_result["changed"] == 0 else "fewer"
    if d_prediction.value is None:
        part_d = mo.vstack([d_prediction, locked(
            "Batch-invariant kernel: the reduction is always split into 4 pieces, whatever the batch size."
        )])
    else:
        _rows = "\n".join(f"| {b} | {t:,} | {i:,} |" for b, (t, i) in d_extra.items())
        part_d = mo.vstack([
            d_prediction,
            d_batch,
            mo.md(
                f"""
    **{d_result['changed']} of {d_result['users']} users** get different text at batch {d_batch.value}.
    Bits of all 64 outputs: alone `{d_result['digest_alone']}`, batched `{d_result['digest_batched']}`.

    **The price.** A fixed split cannot adapt. Extra additions to combine partial sums in one 128-output
    layer:

    | Batch | Adaptive split | Fixed split |
    |---|---|---|
    {_rows}
    """
            ),
            verdict(
                d_prediction, _truth,
                "Fix the order and the bits stop depending on the batch. The cost is performance: at small "
                "batches the kernel can no longer pick the split that fills the chip, and at large batches it "
                "does reduction work it does not need. Thinking Machines measured 1,000 requests on Qwen3-8B "
                "at 26 s with vLLM's default kernels and 42 s batch-invariant; all 1,000 completions became "
                "identical. Matrix multiplication is only one kernel: normalisation and attention must be "
                "made batch-invariant too.",
            ),
        ])
    return (part_d,)


@app.cell
def _(e_batch, e_prediction, e_quality, e_result, locked, mo, verdict):
    _truth = "same" if e_result["changed"] == 0 else "differ"
    if e_prediction.value is None:
        part_e = mo.vstack([e_prediction, locked(
            "Weights and activations are quantised to int8; products are summed exactly in int32. The "
            "kernel keeps its adaptive split."
        )])
    else:
        part_e = mo.vstack([
            e_prediction,
            e_batch,
            mo.md(
                f"""
    **{e_result['changed']} of {e_result['users']} users** get different text, with the adaptive split
    still on. Bits: alone `{e_result['digest_alone']}`, batched `{e_result['digest_batched']}`.

    **The price.** The integer model is a different model: its greedy token matches the float model's
    for **{e_quality['token_agreement']:.1%}** of tokens, and **{e_quality['identical_sequences']} of
    {e_quality['users']}** users get exactly the same text.
    """
            ),
            verdict(
                e_prediction, _truth,
                "Integer addition is exact while it does not overflow, so every order gives the same sum and "
                "the kernel keeps its fast, adaptive split. The bits match across batch sizes, and across "
                "chips too: an integer sum is the same on any GPU or CPU. The cost moves from speed to "
                "accuracy, and to the work of quantising well. At real scale this is achievable: samebit runs "
                "Qwen2.5-0.5B in integers within 0.75% of float32 perplexity, with identical bits on an NVIDIA "
                "T4 GPU and an x86 CPU.",
            ),
        ])
    return (part_e,)


@app.cell
def _(mo, part_a, part_b, part_c, part_d, part_e):
    mo.ui.tabs({
        "A · One sum, five orders": part_a,
        "B · The batch picks the kernel": part_b,
        "C · From bits to words": part_c,
        "D · Batch-invariant kernels": part_d,
        "E · Integer arithmetic": part_e,
    })
    return


@app.cell
def _(mo):
    s_rl = mo.ui.radio(
        {"Default fast kernels": "fast", "Batch-invariant kernels": "invariant", "Integer arithmetic": "integer"},
        label="An RL trainer scores the samples its inference server generated, recomputing their "
        "probabilities in a different batch. Which contract would you choose?",
    )
    s_audit = mo.ui.radio(
        {"Default fast kernels": "fast", "Batch-invariant kernels": "invariant", "Integer arithmetic": "integer"},
        label="A regulator must reproduce a model's decision bit for bit, on different hardware, years later.",
    )
    return s_audit, s_rl


@app.cell
def _(mo, s_audit, s_rl):
    _notes = {
        "fast": "Fast, but the trainer and the server can disagree about the same tokens.",
        "invariant": "Same bits on the same kernels and hardware, at a speed cost.",
        "integer": "Same bits on any hardware, at an accuracy cost that must be measured.",
    }
    mo.vstack([
        mo.md("## Synthesis"),
        s_rl,
        mo.md(f"*{_notes[s_rl.value]}*" if s_rl.value else ""),
        s_audit,
        mo.md(f"*{_notes[s_audit.value]}*" if s_audit.value else ""),
        mo.md(
            r"""
    **Key takeaways**

    1. Floating-point addition is not associative, so a reduction's result depends on its order.
    2. Fast kernels choose their order by batch size, so a server's output depends on the batch around
       each request. This, not randomness, is why temperature 0 is not deterministic.
    3. Batch-invariant kernels restore the same bits on the same hardware, at a speed cost.
    4. Integer arithmetic gives the same bits on any hardware, at an accuracy cost.

    Reproducibility is a design choice with a price, like latency or memory. A system should state
    which contract it offers.

    **Read more**

    - Horace He and Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*, 10 September 2025.
      [thinkingmachines.ai](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
    - David Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*, ACM
      Computing Surveys, 1991.
    - samebit: integer LLM inference with identical bits on GPU and CPU.
      [github.com/Shakhtar-Sankur/samebit](https://github.com/Shakhtar-Sankur/samebit)
    """
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
