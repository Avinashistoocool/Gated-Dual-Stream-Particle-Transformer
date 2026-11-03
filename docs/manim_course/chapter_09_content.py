"""Chapter 09 script and isolated numerical validation for Chapters 09--12."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def beat(key, caption, narration, definition, steps, shape="", diagram="flow",
         weights=None, source="Report: Self-attention, derived step by step",
         code=None):
    return dict(key=key, caption=caption, narration=narration, definition=definition,
                steps=steps, shape=shape, diagram=diagram, weights=weights,
                source=source, code=code)


BEATS = [
    beat("recap", "Embeddings turn observations into learned particle vectors.",
         "Previously, we encoded particles and their pairwise relationships as learned vectors. A blue particle vector describes one particle, while an orange relation vector describes a pair. This chapter explains how particles exchange information without assuming an ordering. We first remove gates and pair bias so the underlying attention calculation stays visible.",
         "Queries compare; values carry messages.",
         [r"x_i\in\mathbb R^D", r"h_i=\operatorname{LN}(x_i)",
          r"h_i\longrightarrow(q_i,k_i,v_i)", r"(q_i,k_j)\longrightarrow A_{ij}\longrightarrow o_i"]),
    beat("objective", "You will calculate attention and explain its symmetries.",
         "Our objective is to calculate an entire attention head, including its final message. We will explain why scores are scaled, how heads divide channels, and how padding is excluded. We will then reorder particles and check what must change together. Three solved exercises will test the arithmetic, masking, and shape reasoning.",
         "One comparison becomes one coefficient in a message.",
         [r"\text{compare}\longrightarrow\text{scale}", r"\text{scale}\longrightarrow\text{normalize}",
          r"\text{normalize}\longrightarrow\text{combine}", r"\text{combine}\longrightarrow\text{check}"]),
    beat("axes", "Define every axis before multiplying tensors.",
         "Capital B counts jets in a batch, and N counts padded particle slots. Capital D counts particle channels; H counts attention heads, each containing little d channels. The indices i and j select the receiving query and sending key, while c selects a channel. We suppress the batch and head indices for one small calculation.",
         "i receives; j supplies; c selects a channel.",
         [r"B:\text{jets},\quad N:\text{slots}", r"D=Hd,\quad H:\text{heads}",
          r"X\in\mathbb R^{B\times N\times D}", r"Q,K,V\in\mathbb R^{B\times H\times N\times d}"],
         r"D=Hd"),
    beat("qkv", "Queries and keys choose weights; values supply messages.",
         "The normalized particle state h is a learned vector, not a hand-selected physics attribute. Three separate affine projections produce a query, a key, and a value; each weight matrix maps D channels to D channels before splitting heads. The query asks how to compare, and the key supplies the matching description. The value carries the information to be combined after those comparisons.",
         "W and b are learned projection weights and biases.",
         [r"h_i\in\mathbb R^D", r"q_i=W_Qh_i+b_Q",
          r"k_i=W_Kh_i+b_K", r"v_i=W_Vh_i+b_V"],
         r"W_Q,W_K,W_V:(D,D);\quad b_Q,b_K,b_V:(D,)"),
    beat("rows", "Stored particle rows require transposed projection weights.",
         "Our mathematical vectors are columns, but PyTorch stores particles along tensor rows. The same affine operation therefore multiplies the row table by the transpose of the weight matrix. A column of ones repeats the bias for every particle, rather than introducing a new particle feature. Suppressing the batch axis, both the input and projected tables have N rows and D columns.",
         "The vector of ones broadcasts the bias across particles.",
         [r"X:(N,D),\quad W_Q:(D,D)", r"XW_Q^\top:(N,D)",
          r"\mathbf1_Nb_Q^\top:(N,D)", r"Q=XW_Q^\top+\mathbf1_Nb_Q^\top"],
         r"\mathbf1_N:(N,1);\quad b_Q^\top:(1,D)"),
    beat("dot", "A dot product adds matching channel products.",
         "Let q equal one, zero, and let our two keys be one, zero and zero, one. Each vector has two channels, so d equals two; the dot product returns one scalar per key. For the first key, one times one plus zero times zero gives one. For the second key, one times zero plus zero times one gives zero.",
         "These illustrative vectors are not trained particle measurements.",
         [r"q=(1,0),\quad k_1=(1,0),\quad k_2=(0,1)",
          r"q^\top k_j=q_1k_{j1}+q_2k_{j2}",
          r"q^\top k_1=1(1)+0(0)=1",
          r"q^\top k_2=1(0)+0(1)=0"], r"q,k_1,k_2:(2,);\quad q^\top k_j:()"),
    beat("scale", "Scaling controls the initial spread of content scores.",
         "Define C as the content score before any bias or mask. Divide each dot product by the square root of the head width, which is the square root of two here. The first score becomes approximately zero point seven zero seven one, and the second remains zero. This scaling does not change their order, but it does change how sharply softmax separates them.",
         "C denotes a content score, not a probability.",
         [r"C_j=\frac{q^\top k_j}{\sqrt d}",
          r"(C_1,C_2)=\left(\frac1{\sqrt2},\frac0{\sqrt2}\right)",
          r"\sqrt2\simeq1.4142", r"(C_1,C_2)\simeq(0.7071,0)"],
         r"C:(2,)"),
    beat("variance", "The square-root rule follows an explicit independence assumption.",
         "Assume query and key entries are independent, zero mean, and unit variance, with independent products across channels. Each channel product then has variance one, so adding d products gives variance d. Multiplying a random variable by a constant multiplies its variance by the square of that constant. Division by square root d therefore restores variance one; trained correlated channels need not obey these assumptions exactly.",
         "Variance measures spread; independence is an assumption here.",
         [r"\operatorname{Var}(q_ck_c)=1",
          r"\operatorname{Var}\left(\sum_{c=1}^dq_ck_c\right)=d",
          r"\operatorname{Var}(aZ)=a^2\operatorname{Var}(Z)",
          r"\operatorname{Var}\left(\frac{\sum_cq_ck_c}{\sqrt d}\right)=\frac dd=1"],
         diagram="none"),
    beat("softmax", "Softmax normalizes across keys for one fixed query.",
         "Call the complete pre-softmax scores S, and call the normalized attention weights A. With no bias and no padding, S equals C, so exponentiating gives approximately two point zero two eight one and one. Their sum is approximately three point zero two eight one. Divide each exponential by that same sum to obtain weights zero point six six nine eight and zero point three three zero two.",
         "A row sums to one before training dropout.",
         [r"S=C,\qquad A_j=\frac{e^{S_j}}{\sum_t e^{S_t}}",
          r"(e^{S_1},e^{S_2})\simeq(2.0281,1)",
          r"Z=e^{S_1}+e^{S_2}\simeq3.0281",
          r"(A_1,A_2)\simeq(0.6698,0.3302)"],
         r"S,A:(2,);\quad Z:()", "attention",
         [[.5,.5],[.5,.5],[.5,.5],[.66976155,.33023845]]),
    beat("stable", "Subtracting a common row maximum leaves weights unchanged.",
         "Exponentials can overflow if logits are very large, even when their differences are moderate. Let u denote the largest logit in the current row, and subtract u from every entry. The common exponential factor cancels between numerator and denominator, so the probabilities do not change. This is a numerical stability operation, not a learned attention bias favoring one particle.",
         "u is the maximum score in the current row.",
         [r"u=\max_j S_j", r"A_j=\frac{e^{S_j-u}}{\sum_t e^{S_t-u}}",
          r"S-u=(0,-0.7071)", r"\operatorname{softmax}(S-u)=\operatorname{softmax}(S)"],
         diagram="attention", weights=[[.66976155,.33023845]]*4),
    beat("values", "Values are weighted messages, not the matching scores.",
         "Choose value vectors v one equal to two, zero, and v two equal to zero, four. Both have two channels, just like this head's queries and keys, but they serve a different role. Multiply the first value by its attention weight, obtaining approximately one point three three nine five, zero. Multiply the second value by its weight, obtaining approximately zero, one point three two one zero.",
         "o denotes the resulting message vector.",
         [r"v_1=(2,0),\quad v_2=(0,4)",
          r"o=A_1v_1+A_2v_2",
          r"A_1v_1\simeq(1.3395,0)", r"A_2v_2\simeq(0,1.3210)"],
         r"v_1,v_2,o:(2,)", "attention", [[.66976155,.33023845]]*4),
    beat("message", "Add weighted values channel by channel.",
         "Now add the two weighted messages rather than multiplying the values together. The first output channel is approximately one point three three nine five, and the second is one point three two one zero. Before training dropout, nonnegative weights summing to one make this a convex combination of the value vectors. The later output projection and residual addition do not have to remain inside that convex region.",
         "The message is a vector; attention weights are scalars.",
         [r"o=(1.3395,0)+(0,1.3210)",
          r"o\simeq(1.3395,1.3210)",
          r"A_1+A_2=1,\quad A_j\ge0",
          r"o\in\operatorname{conv}\{v_1,v_2\}"],
         diagram="attention", weights=[[.66976155,.33023845]]*4),
    beat("code", "Run the complete calculation in PyTorch.",
         "This code is executed with PyTorch, and the displayed numbers come from its captured standard output. Matrix multiplication compares q with both key rows, and square root two scales the scores. Softmax normalizes the final key axis, after which matrix multiplication combines the value rows. The assertions compare full precision results with the hand calculation, allowing only the displayed rounding error.",
         "Printed numbers are executed, not illustrative placeholders.",
         [], diagram="none", code="attention"),
    beat("matrix", "Every query gets its own normalized attention row.",
         "So far we followed one query; self-attention repeats the same calculation for every particle. Multiplying the query matrix by the transposed key matrix creates an N by N score table for each head. Row i contains every key comparison for query i, and softmax normalizes that row independently. Multiplying the resulting attention matrix by V produces one d-channel message per query.",
         "Rows receive messages; columns supply values.",
         [r"Q,K,V:(N,d)", r"S=\frac{QK^\top}{\sqrt d}:(N,N)",
          r"A=\operatorname{softmax}_{j}(S):(N,N)", r"O=AV:(N,d)"],
         r"\text{one batch, one head}", "matrix"),
    beat("mask", "Padding must be excluded before row normalization.",
         "Let m j be one for a real particle and zero for padding. Define the additive mask M as zero for valid keys and negative infinity for padded keys, independently of the query row. Exponentiating negative infinity gives zero, so a padded value receives no attention mass. With only our first key valid, its weight becomes one and the output becomes two, zero.",
         "m marks valid keys; M is an additive logit mask.",
         [r"m=(1,0),\quad M=(0,-\infty)",
          r"S=C+M=(0.7071,-\infty)", r"A=(1,0)", r"o=1(2,0)+0(0,4)=(2,0)"],
         r"m:(B,N);\quad M:(B,1,1,N)", "attention",
         [[.66976155,.33023845],[.66976155,.33023845],[1,0],[1,0]],
         source="Report self-attention; gds_part_colab.py GDSParT._pair_inputs"),
    beat("mask_mistake", "Zeroing padded values does not remove their normalization mass.",
         "Suppose we replace the padded value by zero but leave its finite score untouched. Softmax still allocates approximately one third of the row to that padded key, shrinking the real message incorrectly. Also, masking keys does not automatically zero padded query outputs; those rows must remain excluded from later pooling or reads. Every jet should contain at least one valid key, because an all-invalid softmax row is not a valid probability distribution.",
         "A mask changes normalization, not merely the stored value.",
         [r"v_2=0,\quad M_2=0\quad\text{(wrong)}",
          r"o_{\rm wrong}=0.6698(2,0)\simeq(1.3395,0)",
          r"M_2=-\infty\quad\Longrightarrow\quad o=(2,0)",
          r"\sum_jm_j>0"],
         diagram="attention", weights=[[.66976155,.33023845],[.66976155,.33023845],[1,0],[1,0]]),
    beat("mask_api", "Boolean mask conventions depend on the PyTorch operation.",
         "The local particle-attention code uses scaled dot product attention, where a true Boolean mask entry means keep that key. It broadcasts the valid-particle mask across heads and query rows, or inserts negative infinity into the additive bias tensor. MultiheadAttention uses the opposite convention for its key padding mask, where true means exclude. The classifier's class-attention code performs that inversion; copying a mask between APIs without checking semantics changes the model.",
         "Report versus code difference: API masks need explicit conventions.",
         [r"m_j=\mathrm{True}\Longleftrightarrow\text{valid}",
          r"M_{\rm SDPA}=m[:,\mathrm{None},\mathrm{None},:]",
          r"M_{\rm additive,j}=\begin{cases}0&m_j\\-\infty&\neg m_j\end{cases}",
          r"\text{MHA key-padding mask}=\neg m"],
         source="Report compact attention notation versus gds_part_colab.py _pair_inputs and ClassAttentionBlock"),
    beat("heads", "Heads split channels, not the list of particles.",
         "Take the reported channel width D equal to one hundred twenty eight and H equal to eight heads. Each head receives sixteen channels but still sees all N particles and every valid key. A reshape followed by a transpose changes the stored axes from batch, particle, channel to batch, head, particle, head-channel. Each head therefore builds its own N by N attention table rather than processing a disjoint group of particles.",
         "Each head sees every valid particle.",
         [r"D=128,\quad H=8,\quad d=D/H=16",
          r"(B,N,128)\longrightarrow(B,N,8,16)",
          r"(B,N,8,16)\longrightarrow(B,8,N,16)",
          r"A:(B,8,N,N)"],
         r"D=Hd", "heads"),
    beat("merge", "Concatenation restores channels before another learned projection.",
         "After each head combines its values, concatenate the head messages along the channel axis. This restores D channels for each particle without averaging the heads into one message. The output affine projection then mixes those concatenated channels, and the surrounding block later adds a residual connection. No architectural rule forces a specific head to represent angles or momentum; such interpretations require measured evidence.",
         "W sub O mixes the concatenated head channels.",
         [r"O^{(h)}:(B,N,d)", r"O_{\rm cat}=[O^{(1)}|\cdots|O^{(H)}]:(B,N,D)",
          r"Y=O_{\rm cat}W_O^\top+\mathbf1b_O^\top",
          r"X_{\rm next}=X+Y"],
         r"W_O:(D,D);\quad b_O:(D,)", "heads"),
    beat("permutation", "Reordering particles must reorder both attention axes.",
         "Let P be a permutation matrix that reorders the particle rows without changing their measurements. Because the same projections act on each row, Q, K, and V all acquire that same row permutation. The score matrix transforms as P S P transpose, which reorders both receivers and senders. Row softmax respects this joint reordering, so the output becomes P times the original output.",
         "P reorders particles; it does not change their features.",
         [r"X\mapsto PX,\quad Q,K,V\mapsto PQ,PK,PV",
          r"S\mapsto PSP^\top", r"A\mapsto PAP^\top",
          r"(PAP^\top)(PV)=PAV"],
         r"P:(N,N);\quad P^\top P=I", "matrix"),
    beat("equivariance", "Particle outputs reorder; a jet prediction should not.",
         "Equivariance means that particle outputs move with the reordered particle inputs, not that every output row stays fixed. An invariant jet summary, such as a valid masked mean or a class-token read, removes that ordering dependence. Pair tensors must be reordered on both particle axes, and masks must follow the particles too. This statement assumes no absolute position-index embeddings and is tested in evaluation mode without random dropout.",
         "Equivariance concerns rows; invariance concerns the whole jet.",
         [r"F(PX)=PF(X)", r"R_{ij}\mapsto R_{\pi(i)\pi(j)}",
          r"m_i\mapsto m_{\pi(i)}", r"\operatorname{pool}(PF(X))=\operatorname{pool}(F(X))"],
         diagram="matrix"),
    beat("exercise1", "Exercise one: equal scores produce an average message.",
         "First exercise: keep the two values two, zero and zero, four, but set the query to zero. Both dot products are zero, so both exponentials equal one regardless of the square-root scaling. Normalizing gives one half to each key, not zero attention everywhere. The message is one half times each value, which adds to one, two.",
         "A zero query still reads values through normalized weights.",
         [r"q=(0,0)\Longrightarrow S=(0,0)", r"e^S=(1,1)",
          r"A=(1/2,1/2)", r"o=\tfrac12(2,0)+\tfrac12(0,4)=(1,2)"],
         diagram="attention", weights=[[.5,.5]]*4),
    beat("exercise2", "Exercise two: mask the stronger key before softmax.",
         "Second exercise: return to scores zero point seven zero seven one and zero, but now mask the first key. Add negative infinity to the first score, leaving the second score finite. The normalized weights become zero and one even though the unmasked first content score was larger. The output is therefore exactly the second value, zero, four; padding validity wins over content preference.",
         "A masked key cannot win through a large content score.",
         [r"C=(0.7071,0),\quad M=(-\infty,0)",
          r"S=(-\infty,0)", r"A=(0,1)", r"o=0(2,0)+1(0,4)=(0,4)"],
         diagram="attention", weights=[[.66976155,.33023845],[.66976155,.33023845],[0,1],[0,1]]),
    beat("exercise3", "Exercise three: track every head and score axis.",
         "Third exercise: a batch has two jets, three particle slots, eight channels, and two heads. Dividing eight channels by two heads gives four channels per head, so queries have shape two, two, three, four. Multiplying by transposed keys produces scores with shape two, two, three, three. Weighted values return two, two, three, four, and merging heads restores two, three, eight.",
         "The score table has two particle axes, not a channel axis.",
         [r"B=2,\ N=3,\ D=8,\ H=2,\ d=4",
          r"Q,K,V:(2,2,3,4)", r"S,A:(2,2,3,3)",
          r"AV:(2,2,3,4)\longrightarrow Y:(2,3,8)"],
         diagram="heads"),
    beat("gds", "GDS keeps this calculation and changes its matching inputs.",
         "Locate attention on our miniature model map between embeddings and prediction. In GDS, orange pair relations produce red query and key gates, and also a teal additive pair bias. The subsequent score normalization and value combination remain the operations we just derived. Dual stream refers to the blue particle states and orange relations, not to the query and key projections inside ordinary attention.",
         "Dual stream means X and R, not Q and K.",
         [r"X\longrightarrow Q,K,V", r"R\longrightarrow G^Q,G^K,\mathcal B",
          r"S=\frac{(Q\odot G^Q)(K\odot G^K)^\top}{\sqrt d}+\mathcal B+M",
          r"O=\operatorname{softmax}_{j}(S)V"],
         source="Report GDS gates: Applying the gates"),
    beat("summary", "Attention turns comparisons into weighted messages.",
         "First, queries and keys generate scaled scores, while values provide the messages being combined. Second, masks remove invalid keys before row normalization, and multiple heads repeat the calculation across channel groups. Third, particle outputs are permutation equivariant when every particle-indexed input is reordered consistently. Next we add direct pair bias and calculate exactly how it changes the relative odds.",
         "", [], diagram="none"),
]

SUMMARY = ["Queries and keys set weights; values supply messages.",
           "Masks remove invalid keys before normalization.",
           "Particle outputs reorder consistently with particle inputs."]
CHAPTER = {b["key"]: (b["caption"], b["narration"], b["source"]) for b in BEATS}
CODE = {
    "attention": """import torch
q = torch.tensor([1., 0.])
k = torch.eye(2)
v = torch.tensor([[2., 0.],
                  [0., 4.]])
s = q @ k.T / 2**0.5
a = s.softmax(-1)
print(s)
print(a)
print(a @ v)"""
}


def validate():
    import torch
    torch.set_num_threads(1)
    torch.set_printoptions(precision=4, sci_mode=False)
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    reports = {}
    for number in range(9, 13):
        module = importlib.import_module(f"chapter_{number:02}_content")
        for item in module.BEATS:
            for sentence in (item["caption"], item["definition"], *module.SUMMARY):
                if sentence:
                    assert len(sentence.split()) <= 12, sentence
                    assert sentence.endswith((".", "?")), sentence
            sentences = re.split(r"(?<=[.!?])\s+", item["narration"])
            assert len(sentences) >= max(1, len(item["steps"])), item["key"]
            assert all(len(s.split()) <= 36 for s in sentences), item["key"]
        outputs = {}
        for key, source in module.CODE.items():
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(compile(source, f"chapter_{number:02}_{key}", "exec"), {})
            outputs[key] = stream.getvalue()
        (out / f"chapter_{number:02}_executed_examples.json").write_text(
            json.dumps(outputs, indent=2), encoding="utf-8")
        reports[str(number)] = module.verify(torch)
        reports[str(number)]["words"] = sum(len(b["narration"].split()) for b in module.BEATS)
        reports[str(number)]["blocks"] = len(module.BEATS)
    (out / "chapter_09_12_validation.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


def verify(torch):
    q = torch.tensor([1., 0.], dtype=torch.float64)
    k = torch.eye(2, dtype=torch.float64)
    v = torch.tensor([[2., 0.], [0., 4.]], dtype=torch.float64)
    s = q @ k.T / math.sqrt(2)
    a = s.softmax(-1)
    assert torch.allclose(s, torch.tensor([.7071, 0.], dtype=s.dtype), atol=5e-5, rtol=0)
    assert torch.allclose(a, torch.tensor([.6698, .3302], dtype=s.dtype), atol=5e-5, rtol=0)
    assert torch.allclose(a @ v, torch.tensor([1.3395, 1.3210], dtype=s.dtype), atol=5e-5, rtol=0)
    for mask, expected in [([True, False], [2., 0.]), ([False, True], [0., 4.])]:
        weights = s.masked_fill(~torch.tensor(mask), -torch.inf).softmax(-1)
        assert torch.equal(weights @ v, torch.tensor(expected, dtype=v.dtype))
    assert torch.equal(torch.zeros(2, dtype=v.dtype).softmax(-1) @ v,
                       torch.tensor([1., 2.], dtype=v.dtype))
    torch.manual_seed(9)
    Q, K, V = [torch.randn(2, 2, 3, 4, dtype=v.dtype) for _ in range(3)]
    mask = torch.tensor([[True, True, False], [True, True, True]])
    perm = torch.tensor([2, 0, 1])
    def run(Q, K, V, m):
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=m[:, None, None, :])
    result = run(Q, K, V, mask)
    permuted = run(Q[:, :, perm], K[:, :, perm], V[:, :, perm], mask[:, perm])
    assert torch.allclose(permuted, result[:, :, perm], atol=1e-12)
    return {"passed": ["scaled logits", "softmax", "weighted values", "both padding masks",
                       "zero-query exercise", "multihead shapes", "masked permutation equivariance"],
            "logits": s.tolist(), "weights": a.tolist(), "output": (a @ v).tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate()
