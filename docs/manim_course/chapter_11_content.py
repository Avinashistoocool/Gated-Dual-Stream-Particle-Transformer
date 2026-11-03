"""Relational gate construction with production-equivalence numerical checks."""
from chapter_09_content import beat

SOURCE = "Report: GDS gates, full construction; gds_part_colab.py RelationalGate and ParticleAttentionBlock"
BEATS = [
    beat("recap", "Relations can alter attention through more than additive bias.",
         "Previously, we added a pair-specific score before softmax and calculated its exact effect on relative odds. The blue particle stream supplied learned queries, keys, and values, while orange relations supplied the teal bias. GDS adds a second relation route that rescales query and key channels before comparison. This chapter constructs those gates rather than treating them as unexplained switches.",
         "Dual stream means particle states X and pair relations R.",
         [r"X\longrightarrow Q,K,V", r"R\longrightarrow\mathcal B",
          r"R\longrightarrow G^Q,G^K", r"S=C_{\rm gated}+\mathcal B+M"],
         diagram="gates"),
    beat("objective", "You will build and test each gate-construction step.",
         "Our objective is to reduce a relation row to a particle summary, then turn that summary into separate query and key gates. We will compare the reported shared masked mean with the default code's state-conditioned summary. We will calculate identity initialization and explain exactly what it does not switch off. Worked exercises and executed production checks will verify masking, shapes, and gate application.",
         "A gate is a vector of channel scales.",
         [r"R\longrightarrow s_i", r"s_i\longrightarrow z_i^Q,z_i^K",
          r"z_i^Q,z_i^K\longrightarrow g_i^Q,g_i^K", r"(q_i,k_i)\longrightarrow(q_i',k_i')"],
         diagram="gates"),
    beat("symbols", "A summary removes the neighbour axis, not the channel axis.",
         "Capital B counts jets, N counts padded slots, D sub p counts relation channels, and D counts particle channels. The relation entry r i j has D sub p channels and describes receiver i paired with neighbour j. Let m j be one for a valid neighbour and zero for padding, and let n count valid neighbours. The summary s i retains the receiver address and relation channels while reducing the neighbour address.",
         "s is a relation summary; S is an attention logit.",
         [r"R:(B,N,N,D_p),\quad m:(B,N)",
          r"r_{ij}\in\mathbb R^{D_p},\quad n=\sum_jm_j",
          r"s_i\in\mathbb R^{D_p}",
          r"G^Q,G^K:(B,N,D)"],
         r"D_p=64,\quad D=128", "gates"),
    beat("mean", "The shared summary averages only valid relation neighbours.",
         "Hold receiver i fixed and walk across its orange relation row. Multiply every relation vector by the corresponding validity mask, add those masked vectors, and divide by the number of valid neighbours. The denominator counts neighbours, not relation channels, heads, or batch examples. For a nonempty jet this masked mean returns one D sub p channel vector per receiver.",
         "The sum reduces j while retaining i and relation channels.",
         [r"r_{i1},r_{i2},\ldots,r_{iN}",
          r"\sum_{j=1}^Nm_jr_{ij}", r"n=\sum_{j=1}^Nm_j>0",
          r"s_i=\frac{\sum_jm_jr_{ij}}{n}"],
         diagram="gates", source=SOURCE),
    beat("mean_numbers", "Padding must not dilute the relation summary.",
         "Use a relation row containing vectors two, zero; zero, four; and one hundred, one hundred. The validity mask is one, one, zero, so the final stored vector must contribute nothing despite its large numbers. The numerator becomes two, four, and the denominator is two valid neighbours. Dividing gives summary one, two; dividing by three padded slots would incorrectly shrink both channels.",
         "The padded numbers are arbitrary and must have no effect.",
         [r"r_{i,:}=[(2,0),(0,4),(100,100)]",
          r"m=(1,1,0)\Longrightarrow\sum_jm_jr_{ij}=(2,4)",
          r"n=1+1+0=2", r"s_i=(2,4)/2=(1,2)"],
         r"r_{i,:}:(3,2);\quad s_i:(2,)", "gates"),
    beat("broadcast", "Broadcast the validity mask onto the neighbour axis.",
         "The implementation stores the mask with batch and neighbour axes, then inserts singleton receiver and channel axes. Multiplication broadcasts that mask over every receiver and relation channel without changing which neighbours are valid. Summing axis two removes the neighbour axis, and the broadcast denominator divides each receiver summary by the same valid-neighbour count. The code clamps the denominator to at least one, but meaningful attention still requires nonempty jets.",
         "Report versus code difference: empty denominators are clamped.",
         [r"m:(B,N)\longrightarrow(B,1,N,1)",
          r"R\odot m:(B,N,N,D_p)",
          r"\operatorname{sum}_{j}(R\odot m):(B,N,D_p)",
          r"s=\frac{\operatorname{sum}_{j}(R\odot m)}{\max(1,\sum_jm_j)}"],
         diagram="gates", source=SOURCE),
    beat("diagonal", "The actual aggregation includes valid self-pair embeddings.",
         "The raw self-pair feature vector U i i is zero, but its embedded relation need not be zero. Biases in the pair embedding can turn that zero input into a nonzero learned self relation. The gate's mask excludes padded neighbours, not the diagonal neighbour j equal to i. Therefore the self-pair participates in both masked-mean and conditioned summaries whenever that particle is valid.",
         "Raw zero self-pairs need not remain zero after biased layers.",
         [r"U_{ii}=0", r"r_{ii}=\operatorname{PairMLP}(0)",
          r"r_{ii}\ne0\ \text{in general}", r"s_i=\frac{m_ir_{ii}+\sum_{j\ne i}m_jr_{ij}}{n}"],
         diagram="gates", source=SOURCE),
    beat("shared", "Shared gates vary across particles but repeat across blocks.",
         "One shared gate module computes a query gate and key gate for every particle before the block loop. Those same particle-specific vectors are reused by every particle block during this forward pass. Shared therefore does not mean one scalar for the jet, nor one identical gate for all particles. The blue particle states still change between blocks while the orange relation tensor and shared gate values are reused.",
         "Shared means across blocks, not across particles.",
         [r"R\longrightarrow(s_1,\ldots,s_N)",
          r"s_i\longrightarrow(g_i^Q,g_i^K)",
          r"G^{Q,(1)}=\cdots=G^{Q,(L)}",
          r"g_i^Q\ne g_j^Q\ \text{in general}"],
         r"\ell:\text{block index};\quad L:\text{block count}", "gates"),
    beat("condition_symbols", "Conditioned summaries read the current normalized particle state.",
         "The default code instead gives each block its own gate module and context projection. For block ell, normalize the current particle state x i using that block's first LayerNorm to obtain h i. A learned context projection maps its D particle channels to D sub p relation channels, producing c i. This context determines which relations matter now, but it does not rewrite the relation tensor itself.",
         "c is a context vector with relation-channel width.",
         [r"h_i^{(\ell)}=\operatorname{LN}_1^{(\ell)}(x_i^{(\ell)})",
          r"W_{\rm ctx}^{(\ell)}:(D_p,D),\quad b_{\rm ctx}^{(\ell)}:(D_p,)",
          r"c_i^{(\ell)}=W_{\rm ctx}^{(\ell)}h_i^{(\ell)}+b_{\rm ctx}^{(\ell)}",
          r"c_i^{(\ell)}\in\mathbb R^{D_p}"],
         diagram="gates", source=SOURCE),
    beat("conditioned", "A small attention read replaces the uniform neighbour average.",
         "Compare each relation r i j with the current context c i by a dot product scaled by square root D sub p. Add the validity mask and normalize across neighbours to obtain coefficients alpha i j, distinct from the main attention weights A. The weighted sum of relation vectors becomes the conditioned summary s i for that block. Every block can read the same R differently as its particle states and gate-module parameters differ.",
         "Alpha reads relations; A combines particle values.",
         [r"e_{ij}^{(\ell)}=\frac{r_{ij}^{\top}c_i^{(\ell)}}{\sqrt{D_p}}+M_{ij}",
          r"\alpha_{ij}^{(\ell)}=\operatorname{softmax}_j(e_{ij}^{(\ell)})",
          r"s_i^{(\ell)}=\sum_j\alpha_{ij}^{(\ell)}r_{ij}",
          r"R^{(1)}=\cdots=R^{(L)}"],
         r"e,\alpha:(B,N,N);\quad s:(B,N,D_p)", "gates"),
    beat("condition_numbers", "Changing context changes the summary without changing relations.",
         "Choose two valid relation vectors one, zero and zero, one, with two relation channels. If context is zero, zero, both comparison scores are zero, so the summary is one half, one half. Change only context to square root two, zero; the scaled scores become one and zero. Softmax then gives approximately zero point seven three one one and zero point two six eight nine, which become the summary coordinates for these basis relations.",
         "The relation vectors stay fixed throughout this calculation.",
         [r"r_{i1}=(1,0),\quad r_{i2}=(0,1),\quad D_p=2",
          r"c_i=(0,0)\Longrightarrow s_i=(0.5,0.5)",
          r"c_i=(\sqrt2,0)\Longrightarrow e_i=(1,0)",
          r"\alpha_i=s_i\simeq(0.7311,0.2689)"],
         diagram="attention", weights=[[.5,.5],[.5,.5],[.73105858,.26894142],[.73105858,.26894142]]),
    beat("mlp", "Separate nonlinear maps produce query and key preactivations.",
         "Let a stand for either the query role Q or the key role K, and let z be the gate preactivation. The first affine map converts D sub p summary channels into D particle channels, followed by the GELU activation. A second affine map returns D preactivations, one per channel to be gated. Query and key maps have separate weights and biases, so a particle can search differently from how it is matched.",
         "The superscript a selects Q or K, not an exponent.",
         [r"a\in\{Q,K\},\quad W_1^a:(D,D_p),\quad b_1^a:(D,)",
          r"u_i^a=W_1^as_i+b_1^a",
          r"t_i^a=\operatorname{GELU}(u_i^a)",
          r"z_i^a=W_2^at_i^a+b_2^a"],
         r"W_2^a:(D,D);\quad z_i^a:(D,)", "gates"),
    beat("activation", "One plus hyperbolic tangent produces positive channel scales.",
         "Apply hyperbolic tangent separately to every gate preactivation z, then add one. For finite real inputs, tanh lies strictly between negative one and one, so gate values lie between zero and two. A zero preactivation gives an identity scale of one; negative preactivations suppress and positive ones amplify that coordinate. The gates are not probabilities, do not sum to one, and do not directly gate the value projection.",
         "Finite-precision saturation may numerically reach the endpoint values.",
         [r"g_i^a=\mathbf1+\tanh(z_i^a)",
          r"-1<\tanh(z)<1\Longrightarrow0<g<2",
          r"z=0\Longrightarrow g=1",
          r"z<0\Longrightarrow g<1;\quad z>0\Longrightarrow g>1"],
         diagram="gates", source=SOURCE),
    beat("activation_numbers", "Invert tanh to construct exact illustrative gate scales.",
         "Choose a preactivation equal to negative hyperbolic arctangent of one half for the first channel, and zero for the second. Hyperbolic tangent of that first value is negative one half, so adding one gives gate one half. The zero second preactivation gives gate one, producing vector one half, one. Reversing the sign of the first preactivation gives one and one half, demonstrating amplification without changing the activation rule.",
         "Arctanh is the inverse of tanh inside its valid interval.",
         [r"z=(-\operatorname{arctanh}(1/2),0)",
          r"\tanh(z)=(-1/2,0)",
          r"g=1+\tanh(z)=(1/2,1)",
          r"1+\tanh(\operatorname{arctanh}(1/2))=3/2"],
         diagram="gates"),
    beat("identity", "Zero final-layer parameters initialize the gate path to identity.",
         "In identity mode, the code sets the final gate-layer weight matrix and bias vector to zero, separately for both gate networks. Whatever summary and hidden activation enter that final layer, every preactivation z is therefore zero. Applying one plus tanh produces exactly one in every query and key gate channel. This initializes only the multiplicative gate path to identity; the enabled additive pair bias still changes attention.",
         "Identity gates do not remove the hybrid model's pair bias.",
         [r"W_2^Q=W_2^K=0,\quad b_2^Q=b_2^K=0",
          r"z_i^Q=z_i^K=0",
          r"g_i^Q=g_i^K=\mathbf1",
          r"S_{\rm init}=QK^\top/\sqrt d+\mathcal B+M"],
         diagram="gates", source=SOURCE),
    beat("init_gradient", "Identity initialization does not permanently freeze gate learning.",
         "The derivative of one plus tanh at zero is one, so the final affine layer can receive gradients immediately. Its weight gradient depends on the preceding hidden activation, while its bias gradient can also be nonzero. At the initial step, the zero final weight matrix blocks the gate-path gradient to earlier gate layers. Once the final matrix changes, those earlier layers can receive gradients through this path as well.",
         "c selects output; r selects the preceding hidden channel.",
         [r"\left.\frac{dg}{dz}\right|_{z=0}=1",
          r"\left.\frac{\partial g_c}{\partial(W_2)_{cr}}\right|_{\rm init}=t_r",
          r"\left.\frac{\partial g_c}{\partial(b_2)_c}\right|_{\rm init}=1",
          r"\left.\frac{\partial z}{\partial t}\right|_{\rm init}=W_2=0"],
         diagram="gates", source=SOURCE),
    beat("sigmoid", "Sigmoid mode is a different initialization and range.",
         "The older sigmoid mode maps preactivations into the interval between zero and one rather than zero and two. A sigmoid of zero is one half, but this code does not zero-initialize its final gate layer in sigmoid mode. Consequently sigmoid gates are not guaranteed to start at exactly one half. Do not copy identity-mode initialization claims into the older mode simply because both use a bounded activation.",
         "A sigmoid equals one half only when its input is zero.",
         [r"g_{\rm sigmoid}=\sigma(z)=\frac1{1+e^{-z}}",
          r"0<g_{\rm sigmoid}<1",
          r"\sigma(0)=1/2",
          r"z_{\rm random}\ne0\Longrightarrow g_{\rm init}\ne1/2\ \text{in general}"],
         diagram="gates", source=SOURCE),
    beat("apply", "Gate learned query and key coordinates before splitting heads.",
         "The implementation first projects the normalized particle states into full D-channel queries and keys. It then multiplies each projected coordinate by its matching gate coordinate, preserving the D-channel shape. Only after that elementwise multiplication does it split queries and keys into heads. Values are projected and split but remain directly ungated; changes in attention still alter their final weighted combination.",
         "Red gates multiply learned coordinates, not raw detector features.",
         [r"q_i=W_Qh_i+b_Q,\quad k_i=W_Kh_i+b_K",
          r"q_i'=q_i\odot g_i^Q,\quad k_i'=k_i\odot g_i^K",
          r"(B,N,D)\longrightarrow(B,H,N,d)",
          r"v_i'=v_i"],
         r"q_i,k_i,g_i^Q,g_i^K:(D,)", "gates", source=SOURCE),
    beat("apply_numbers", "A half-strength query coordinate changes the attention weights.",
         "Reuse query one, zero and the two basis keys, with both key gates equal to one. Set the query gate to one half, one, so the gated query becomes one half, zero. With no additive bias, the logits are one over twice square root two and zero, approximately zero point three five three six and zero. Their softmax is approximately zero point five eight seven five and zero point four one two five, closer to equal weighting than before.",
         "This example changes the gate while holding pair bias at zero.",
         [r"q=(1,0),\quad g^Q=(0.5,1),\quad g_1^K=g_2^K=(1,1)",
          r"q'=q\odot g^Q=(0.5,0)",
          r"S=(0.5/\sqrt2,0)\simeq(0.3536,0)",
          r"A\simeq(0.5875,0.4125)"],
         diagram="attention", weights=[[.66976155,.33023845],[.66976155,.33023845],[.587479,.412521],[.587479,.412521]]),
    beat("code", "Run masking, identity initialization, and gated attention.",
         "The first executed calculation excludes the padded relation and prints the masked summary one, two. The next calculation applies one plus tanh to a zero vector and prints identity gates. Finally, the code gates the query, computes its scaled comparisons, and prints the normalized attention weights. Separate assertions also instantiate the production gate class and compare its summaries and initialized outputs with independently calculated expectations.",
         "Production checks verify these executed calculations.",
         [], diagram="none", code="gates"),
    beat("counts", "Separate query and key networks have separate parameter costs.",
         "Each gate network has a sixty four to one hundred twenty eight affine layer and a one hundred twenty eight to one hundred twenty eight affine layer. Include both sets of biases, then double the total because query and key maps do not share parameters. This gives forty nine thousand six hundred sixty four parameters for the shared gate module. A conditioned module adds eight thousand two hundred fifty six context-projection parameters, giving fifty seven thousand nine hundred twenty per block.",
         "Sharing and separate blocks have different parameter costs.",
         [r"D_pD+D+D^2+D=24{,}832",
          r"2(24{,}832)=49{,}664",
          r"D_pD+D_p=8{,}256",
          r"49{,}664+8{,}256=57{,}920"],
         diagram="gates", source=SOURCE),
    beat("exercise1", "Exercise one: recompute a summary after masking one neighbour.",
         "First exercise: use the same relation row two, zero; zero, four; and one hundred, one hundred, but keep only the second neighbour valid. The mask is zero, one, zero, so the numerator is exactly zero, four. There is only one valid neighbour, hence dividing by the count leaves zero, four unchanged. A shared module can then produce different gates from this new summary, but identity-initialized gates remain one until their final layers learn.",
         "Masking changes summaries despite identical initialized gates.",
         [r"m=(0,1,0)", r"\sum_jm_jr_{ij}=(0,4)",
          r"s_i=(0,4)/1=(0,4)", r"W_2=b_2=0\Longrightarrow g_i=\mathbf1"],
         diagram="gates"),
    beat("exercise2", "Exercise two: derive a conditioned summary from two scores.",
         "Second exercise: retain basis relation vectors one, zero and zero, one, but choose context zero, square root two. The scaled relation comparisons become zero and one, reversing the earlier example. Their softmax is approximately zero point two six eight nine and zero point seven three one one. Weighting the basis relations therefore gives exactly those two summary coordinates, while the original relation tensor remains unchanged.",
         "Changing context reverses the read, not the stored relation row.",
         [r"c_i=(0,\sqrt2),\quad D_p=2",
          r"e_i=(0,1)", r"\alpha_i\simeq(0.2689,0.7311)",
          r"s_i=0.2689(1,0)+0.7311(0,1)"],
         diagram="attention", weights=[[.5,.5],[.5,.5],[.26894142,.73105858],[.26894142,.73105858]]),
    beat("configuration", "Reported shared gates and default conditioned gates are different.",
         "The report's named shared v2 run uses one masked-mean gate pair per particle, reused across blocks, together with additive pair bias. Constructing the default configuration instead creates separate state-conditioned gates within each block, still with pair bias. A gates-disabled run retains only the local bias route and is not the complete published Particle Transformer. In all three cases discussed here, the relation tensor is computed once per forward pass rather than recurrently updated.",
         "Report versus code difference: select shared gates explicitly.",
         [r"\text{reported}:\ \texttt{per\_layer\_gate=False}",
          r"\text{default}:\ \texttt{per\_layer\_gate=True}",
          r"\text{local proxy}:\ \texttt{gate\_mode=none}",
          r"R\ \text{is reused across particle blocks}"],
         diagram="gates", source="Report configuration table; gds_part_colab.py GDSParTConfig"),
    beat("summary", "Relation summaries generate separate query and key channel scales.",
         "First, a masked mean or a state-conditioned relation read reduces the neighbour axis without modifying the stored relation tensor. Second, separate query and key networks produce positive channel scales through one plus tanh, initialized to identity by zero final layers. Third, shared gates and conditioned per-block gates describe different configurations. Next we differentiate the resulting scores and test what larger gates really mean for attention.",
         "", [], diagram="none"),
]
SUMMARY = ["Summaries reduce neighbours while preserving relation channels.",
           "Separate query and key gates scale learned coordinates.",
           "Identity gates leave the additive pair bias active."]
for item in BEATS:
    if item["source"] == "Report: Self-attention, derived step by step":
        item["source"] = SOURCE
CHAPTER = {b["key"]: (b["caption"], b["narration"], b["source"]) for b in BEATS}
CODE = {
    "gates": """import torch
r = torch.tensor([[2., 0.], [0., 4.],
                  [100., 100.]])
m = torch.tensor([1., 1., 0.])
print((r*m[:, None]).sum(0)/m.sum())
print(1 + torch.zeros(2).tanh())
q = torch.tensor([1., 0.])
g = torch.tensor([.5, 1.])
print(((q*g) @ torch.eye(2)/2**.5).softmax(0))"""
}


def verify(torch):
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from gds_part_colab import RelationalGate
    torch.manual_seed(11)
    r = torch.tensor([[[[2., 0.], [0., 4.], [100., 100.]],
                       [[1., 3.], [5., 7.], [999., 999.]],
                       [[2., 2.], [4., 4.], [999., 999.]]]])
    mask = torch.tensor([[True, True, False]])
    expected = torch.tensor([[[1., 2.], [3., 5.], [3., 3.]]])
    model = RelationalGate(2, 4)
    seen = []
    handle = model.query_gate.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    gq, gk = model(r, mask)
    handle.remove()
    assert torch.equal(seen[0], expected)
    assert torch.equal(gq, torch.ones_like(gq)) and torch.equal(gk, torch.ones_like(gk))
    assert model.query_gate[0].weight.data_ptr() != model.key_gate[0].weight.data_ptr()
    (gq.sum() + gk.sum()).backward()
    assert torch.count_nonzero(model.query_gate[-1].bias.grad)
    assert torch.count_nonzero(model.query_gate[0].weight.grad) == 0
    with torch.no_grad():
        model.query_gate[-1].weight.fill_(.01)
        model.key_gate[-1].weight.fill_(-.01)
    model.zero_grad()
    learned_q, learned_k = model(r, mask)
    learned_q.sum().backward()
    assert torch.count_nonzero(model.query_gate[0].weight.grad)
    changed_padding = r.clone()
    changed_padding[:, :, 2] = -9999
    unchanged_q, unchanged_k = model(changed_padding, mask)
    assert torch.equal(learned_q, unchanged_q) and torch.equal(learned_k, unchanged_k)
    perm = torch.tensor([2, 0, 1])
    perm_q, perm_k = model(r[:, perm][:, :, perm], mask[:, perm])
    assert torch.equal(perm_q, learned_q[:, perm]) and torch.equal(perm_k, learned_k[:, perm])
    conditioned = RelationalGate(2, 2, conditioned=True)
    with torch.no_grad():
        conditioned.ctx.weight.copy_(torch.eye(2))
        conditioned.ctx.bias.zero_()
    rr = torch.eye(2)[None, None].expand(1, 2, 2, 2).clone()
    xx = torch.tensor([[[2**.5, 0.], [0., 2**.5]]])
    captured = []
    handle = conditioned.query_gate.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    conditioned(rr, torch.ones(1, 2, dtype=torch.bool), xx)
    handle.remove()
    alpha = torch.tensor([[1., 0.], [0., 1.]]).softmax(-1)[None]
    assert torch.allclose(captured[0], alpha, atol=1e-7)
    assert torch.equal(rr, torch.eye(2)[None, None].expand(1, 2, 2, 2))
    g = 1+torch.tensor([-torch.atanh(torch.tensor(.5)), 0.]).tanh()
    assert torch.equal(g, torch.tensor([.5, 1.]))
    weights = (torch.tensor([.5, 0.], dtype=torch.float64)/2**.5).softmax(0)
    assert torch.allclose(weights, torch.tensor([.5875, .4125], dtype=weights.dtype), atol=5e-5, rtol=0)
    assert sum(p.numel() for p in RelationalGate(64, 128).parameters()) == 49664
    assert sum(p.numel() for p in RelationalGate(64, 128, conditioned=True).parameters()) == 57920
    return {"passed": ["production masked summary", "production identity gates", "separate Q/K parameters",
                       "first-step gradients", "later gate-layer gradients", "masked padding invariance",
                       "shared-gate permutation equivariance", "production conditioned summary", "unchanged R",
                       "inverse-tanh gate", "query-gate weights", "shared and conditioned parameter counts"],
            "masked_summary": seen[0].tolist(), "conditioned_summary": captured[0].tolist(),
            "gated_weights": weights.tolist()}
