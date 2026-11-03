"""Pair bias: exact odds, worked messages, and local-versus-published scope."""
from chapter_09_content import beat

SOURCE = "Report: ParT-style attention, an additive relation term; gds_part_colab.py GDSParT._pair_inputs"
BEATS = [
    beat("recap", "Attention compares queries and keys before combining values.",
         "Recall our complete attention calculation: a query one, zero compared with two perpendicular keys gave weights approximately two thirds and one third. Those weights combined two value vectors into a message. The calculation used particle content alone, with no explicit pair relation. Now we ask how a relation can change a comparison even when its content dot product is zero.",
         "Content scores and relation scores contribute different information.",
         [r"C_j=q^\top k_j/\sqrt2", r"C=(0.7071,0)",
          r"A\simeq(0.6698,0.3302)", r"o\simeq(1.3395,1.3210)"],
         diagram="attention", weights=[[.66976155,.33023845]]*4),
    beat("objective", "You will calculate how pair bias changes attention odds.",
         "Our objective is to derive an additive pair bias and explain its multiplicative effect on unnormalized attention weights. We will reuse exactly the same query, keys, and values so the only change is the relation term. We will calculate new probabilities and a new message, then solve two exercises. Finally, we will distinguish this repository's bias path from the full published Particle Transformer.",
         "Keep content fixed while changing only the pair bias.",
         [r"C\longrightarrow C+\mathcal B", r"e^C\longrightarrow e^Ce^{\mathcal B}",
          r"A\longrightarrow A_{\rm biased}", r"o\longrightarrow o_{\rm biased}"], diagram="bias"),
    beat("symbols", "A pair representation supplies one bias per attention head.",
         "Let r i j denote the D sub p channel relation vector for receiver i and sender j. Capital H counts heads, and calligraphic B denotes the additive bias, not batch size B. A learned weight matrix with H rows and D sub p columns maps each relation to H scalar biases. The same projection and bias vector are reused for every pair and every particle block.",
         "Calligraphic B means pair bias; B means batch size.",
         [r"r_{ij}\in\mathbb R^{D_p},\quad D_p=64,\ H=8",
          r"W_{\rm bias}\in\mathbb R^{H\times D_p},\quad b_{\rm bias}\in\mathbb R^H",
          r"\mathcal B_{ij}=W_{\rm bias}r_{ij}+b_{\rm bias}",
          r"\mathcal B:(B,H,N,N)"],
         r"R:(B,N,N,D_p)", "bias", source=SOURCE),
    beat("projection", "Count the projection separately from the entire interaction encoder.",
         "For eight heads and sixty four relation channels, the projection contains eight times sixty four weight entries. There are also eight scalar biases, one for each head. Adding these gives five hundred twenty trainable parameters in this last linear map. This is not a parameter count for the complete relation embedding or the published Particle Transformer interaction encoder.",
         "One projection has 520 parameters, not the whole relation network.",
         [r"\#W_{\rm bias}=HD_p=8(64)=512",
          r"\#b_{\rm bias}=H=8", r"\#\theta_{\rm projection}=512+8=520",
          r"\mathcal B_{ij}^{(h)}=\sum_{c=1}^{64}W_{hc}r_{ijc}+b_h"],
         diagram="bias", source=SOURCE),
    beat("add", "Bias shifts scores before softmax, not after normalization.",
         "For one head, retain the original content scores zero point seven zero seven one and zero. Choose pair biases zero and one, so only the second score receives an additive increase. The resulting logits are zero point seven zero seven one and one; the orthogonal key now has the larger logit. Adding one directly to an already normalized probability would be a different and invalid normalization procedure.",
         "The illustrative bias is chosen, not a trained measurement.",
         [r"C=(1/\sqrt2,0),\quad\mathcal B=(0,1)",
          r"S=C+\mathcal B", r"S=(1/\sqrt2+0,0+1)",
          r"S\simeq(0.7071,1)"], diagram="bias"),
    beat("numbers", "The relation can make the orthogonal key win.",
         "Exponentiate the two new logits to obtain approximately two point zero two eight one and two point seven one eight three. Add them to form the denominator, approximately four point seven four six four. Dividing gives weights approximately zero point four two seven three and zero point five seven two seven. The second key wins despite its zero content dot product, because the bias supplies a separate pair-specific score.",
         "A zero content product does not force a zero biased score.",
         [r"e^S\simeq(2.0281,2.7183)", r"Z\simeq2.0281+2.7183=4.7464",
          r"A=(2.0281/Z,2.7183/Z)", r"A\simeq(0.4273,0.5727)"],
         diagram="attention",
         weights=[[.66976155,.33023845],[.66976155,.33023845],[.4272957,.5727043],[.4272957,.5727043]]),
    beat("message", "The changed probabilities redirect the weighted message.",
         "Keep the original values two, zero and zero, four; pair bias does not directly rewrite either value vector. Multiply the first value by its new weight, giving approximately zero point eight five four six, zero. Multiply the second by its weight, giving approximately zero, two point two nine zero eight. Their sum is the new message, which moves toward the second value solely because its attention coefficient increased.",
         "Unchanged values can produce a changed combined message.",
         [r"v_1=(2,0),\quad v_2=(0,4)",
          r"A_1v_1\simeq(0.8546,0)",
          r"A_2v_2\simeq(0,2.2908)",
          r"o_{\rm biased}\simeq(0.8546,2.2908)"],
         diagram="attention", weights=[[.4272957,.5727043]]*4),
    beat("exponential", "An additive score becomes a multiplicative unnormalized factor.",
         "Use the exponential identity that the exponential of a sum equals the product of exponentials. The numerator for key j factors into its content exponential times its relation exponential. The denominator contains the same product for every valid competing key. Bias therefore multiplies unnormalized weights, but the final normalized probability also depends on every other key's factor.",
         "The denominator couples all valid competitors in a row.",
         [r"A_{ij}=\frac{e^{C_{ij}+\mathcal B_{ij}}}{\sum_t e^{C_{it}+\mathcal B_{it}}}",
          r"e^{C_{ij}+\mathcal B_{ij}}=e^{C_{ij}}e^{\mathcal B_{ij}}",
          r"A_{ij}=\frac{e^{C_{ij}}e^{\mathcal B_{ij}}}{\sum_t e^{C_{it}}e^{\mathcal B_{it}}}",
          r"\mathcal B_{ij}=1\Longrightarrow e^{\mathcal B_{ij}}=e"],
         diagram="bias"),
    beat("odds", "Relative odds cancel the shared normalization denominator.",
         "Compare two valid keys j and k for the same query i, and define their relative odds as A i j divided by A i k. Because both probabilities share a denominator, that denominator cancels in the ratio. Taking a logarithm leaves the difference of content scores plus the difference of pair biases. Only a bias difference changes these relative odds; a common shift cancels.",
         "Relative odds compare two positive weights in the same row.",
         [r"\frac{A_{ij}}{A_{ik}}=\frac{e^{C_{ij}+\mathcal B_{ij}}}{e^{C_{ik}+\mathcal B_{ik}}}",
          r"\frac{A_{ij}}{A_{ik}}=e^{C_{ij}-C_{ik}}e^{\mathcal B_{ij}-\mathcal B_{ik}}",
          r"\log\frac{A_{ij}}{A_{ik}}=C_{ij}-C_{ik}+\mathcal B_{ij}-\mathcal B_{ik}",
          r"\mathcal B_{ij}=\mathcal B_{ik}\Longrightarrow\text{unchanged odds}"],
         diagram="bias"),
    beat("odds_number", "A one-unit bias advantage multiplies relative odds by e.",
         "Initially, the second key's odds relative to the first are exponential negative one over square root two, approximately zero point four nine three one. Giving the second key a one-unit bias advantage multiplies those odds by e, approximately two point seven one eight three. The new odds are approximately one point three four zero three, consistent with the new second-key majority. This does not mean its probability was multiplied by e, because normalization changed too.",
         "Odds multiply by e; normalized probability does not.",
         [r"\frac{A_2}{A_1}=e^{-1/\sqrt2}\simeq0.4931",
          r"\Delta\mathcal B=\mathcal B_2-\mathcal B_1=1",
          r"\frac{A_2'}{A_1'}=e^1e^{-1/\sqrt2}\simeq1.3403",
          r"\frac{0.5727}{0.4273}\simeq1.3403"],
         diagram="attention", weights=[[.66976155,.33023845],[.66976155,.33023845],[.4272957,.5727043],[.4272957,.5727043]]),
    beat("code", "Executed PyTorch verifies probabilities, messages, and relative odds.",
         "The code adds the bias before softmax, using the same two content logits as our hand calculation. Its first output is the new pair of normalized weights, and its second is the weighted value message. The final output divides the new relative odds by the old relative odds, recovering e. These outputs are captured by executing this exact snippet rather than copied from rounded arithmetic.",
         "The code changes bias while holding content and values fixed.",
         [], diagram="none", code="bias"),
    beat("symmetry", "Symmetric pair bias does not require symmetric attention weights.",
         "The same pair encoder acts on every pair, so equal pair features give equal bias outputs rather than arbitrary free table entries. Symmetric pair features in this implementation produce symmetric biases when evaluated by that deterministic shared encoder. Queries and keys have different learned projections, and row normalization uses different denominators. Even a symmetric score matrix can therefore produce an asymmetric attention matrix.",
         "A symmetric score table can have unequal row denominators.",
         [r"\mathcal B_{ij}=\mathcal B_{ji}\ \not\Rightarrow\ A_{ij}=A_{ji}",
          r"S=\begin{bmatrix}0&1\\1&2\end{bmatrix}",
          r"A=\begin{bmatrix}0.2689&0.7311\\0.2689&0.7311\end{bmatrix}",
          r"A_{12}=0.7311\ne0.2689=A_{21}"],
         diagram="matrix"),
    beat("state", "Fixed relation bias can still change evolving particle states.",
         "For a fixed event, the pair embedding R and its bias projection are computed once during the forward pass. They do not read each particle block's current hidden state, unlike state-conditioned gate summaries. Yet changing that bias changes attention weights, which changes messages and therefore subsequent hidden particle states. Independence from the current hidden state is not independence from the event's physical measurements.",
         "R is fixed within one forward pass, not across events.",
         [r"U\longrightarrow R\longrightarrow\mathcal B",
          r"\mathcal B^{(1)}=\cdots=\mathcal B^{(L)}",
          r"A^{(\ell)}=\operatorname{softmax}(C^{(\ell)}+\mathcal B+M)",
          r"A^{(\ell)}\longrightarrow X^{(\ell+1)}"],
         r"\ell:\text{block index};\quad L:\text{block count}", "bias"),
    beat("exercise1", "Exercise one: adding the same bias changes nothing.",
         "First exercise: add three to both logits instead of favoring just one key. Both numerator exponentials acquire the same factor exponential three, and so does their denominator. Cancel that common factor to recover the original probabilities zero point six six nine eight and zero point three three zero two. The message also remains unchanged, so a large common bias need not imply a strong attention preference.",
         "Common offsets cancel; relative offsets matter.",
         [r"\mathcal B=(3,3),\quad S=(0.7071+3,3)",
          r"A_j'=\frac{e^3e^{C_j}}{e^3\sum_t e^{C_t}}",
          r"A'=A\simeq(0.6698,0.3302)",
          r"o'=o\simeq(1.3395,1.3210)"],
         diagram="attention", weights=[[.66976155,.33023845]]*4),
    beat("exercise2", "Exercise two: choose a bias that exactly ties the keys.",
         "Second exercise: keep the first bias zero and find the second bias that makes the two logits equal. The first logit is one over square root two, so set the second bias to exactly that same number. Equal logits produce equal exponentials, hence equal normalized weights of one half. Applying those weights to the unchanged values gives the message one, two.",
         "Solve equality before applying softmax.",
         [r"C=(1/\sqrt2,0),\quad\mathcal B_1=0",
          r"S_1=S_2\Longrightarrow\mathcal B_2=1/\sqrt2",
          r"A=(1/2,1/2)", r"o=\tfrac12(2,0)+\tfrac12(0,4)=(1,2)"],
         diagram="attention", weights=[[.66976155,.33023845],[.66976155,.33023845],[.5,.5],[.5,.5]]),
    beat("scope", "The local bias-only proxy is not the published model.",
         "Original Particle Transformer names a published architecture, not merely one additive term. Disabling gates in this repository while retaining pair bias creates a local bias-only proxy with this repository's embeddings, blocks, and pooling. The reported shared GDS configuration instead retains both gates and bias, while the default code uses conditioned gates inside each block. These distinctions must remain explicit when interpreting comparisons or claiming a reproduction.",
         "Report versus code difference: reported shared gates are not defaults.",
         [r"\text{Published ParT}\ne\text{local bias-only proxy}",
          r"\text{local proxy}:\quad G^Q=G^K=\mathbf1,\ \mathcal B\ne0",
          r"\text{reported shared v2}:\quad\text{shared gates}+\mathcal B",
          r"\text{default v2}:\quad\text{conditioned gates}+\mathcal B"],
         diagram="bias", source="Report model distinctions and configuration table; gds_part_colab.py GDSParTConfig"),
    beat("summary", "Pair bias changes relative scores without rewriting the values.",
         "First, each embedded pair supplies a learned additive score for every attention head. Second, bias differences multiply relative odds, while common row shifts cancel exactly. Third, the local bias-only proxy, the reported shared hybrid, and the published Particle Transformer are distinct systems. Next we build the other relation route: separate query and key gates generated from summaries of relation rows.",
         "", [], diagram="none"),
]
SUMMARY = ["Pair bias adds a relation score before softmax.",
           "Bias differences multiply relative odds, not probabilities.",
           "The local proxy is not the complete published model."]
for item in BEATS:
    if item["source"] == "Report: Self-attention, derived step by step":
        item["source"] = SOURCE
CHAPTER = {b["key"]: (b["caption"], b["narration"], b["source"]) for b in BEATS}
CODE = {
    "bias": """import torch
c = torch.tensor([2**-.5, 0.])
a = (c + torch.tensor([0., 1.])).softmax(0)
v = torch.tensor([[2., 0.], [0., 4.]])
print(a)
print(a @ v)
old = c.softmax(0)
print((a[1]/a[0]) / (old[1]/old[0]))"""
}


def verify(torch):
    c = torch.tensor([2**-.5, 0.], dtype=torch.float64)
    a = (c + torch.tensor([0., 1.], dtype=c.dtype)).softmax(0)
    v = torch.tensor([[2., 0.], [0., 4.]], dtype=c.dtype)
    assert torch.allclose(a, torch.tensor([.4273, .5727], dtype=c.dtype), atol=5e-5, rtol=0)
    assert torch.allclose(a @ v, torch.tensor([.8546, 2.2908], dtype=c.dtype), atol=5e-5, rtol=0)
    old = c.softmax(0)
    ratio = (a[1]/a[0]) / (old[1]/old[0])
    assert torch.allclose(ratio, torch.exp(torch.tensor(1., dtype=c.dtype)))
    assert torch.allclose((c+3).softmax(0), old)
    assert torch.equal((c+torch.tensor([0., 2**-.5], dtype=c.dtype)).softmax(0),
                       torch.tensor([.5, .5], dtype=c.dtype))
    symmetric = torch.tensor([[0., 1.], [1., 2.]], dtype=c.dtype).softmax(-1)
    assert not torch.allclose(symmetric, symmetric.T)
    return {"passed": ["bias weights", "biased message", "odds multiply by e",
                       "constant-shift exercise", "tie exercise", "symmetric scores asymmetric attention"],
            "weights": a.tolist(), "output": (a @ v).tolist(), "odds_multiplier": ratio.item()}
