"""Gate effects: signed derivatives, controlled counterexamples, expressivity."""
from chapter_09_content import beat

SOURCE = "Report: What query gates and key gates actually control; Expressivity; Backpropagation"
BEATS = [
    beat("recap", "Gates rescale learned query and key coordinates before comparison.",
         "Previously, we reduced each orange relation row to a summary and passed it through separate query and key networks. One plus tanh produced red channel scales, with identity initialization setting every scale to one. The gated queries and keys then entered ordinary scaled attention, alongside the retained pair bias. We now ask how changing one scale affects scores, probabilities, and the final message.",
         "A larger gate is not automatically a larger attention weight.",
         [r"g^a=1+\tanh(z^a)", r"q'=q\odot g^Q,\quad k'=k\odot g^K",
          r"S_{ij}=q_i'{}^\top k_j'/\sqrt d+\mathcal B_{ij}+M_{ij}",
          r"A_{i,:}=\operatorname{softmax}(S_{i,:})"], diagram="gates"),
    beat("objective", "You will differentiate gates and test their interpretation.",
         "Our objective is to derive score derivatives for query and key gates, then connect those changes to softmax. We will test a negative-key counterexample where amplifying a key reduces its attention. We will also state the exact conditions under which a gate behaves like temperature, and compare the expressive structure of gates with pair bias. Three solved exercises will make the interpretation falsifiable.",
         "Hold other forward-pass quantities fixed for these partial derivatives.",
         [r"\partial S/\partial g^Q", r"\partial S/\partial g^K",
          r"\partial A/\partial S", r"\text{structure}\ne\text{causal interpretation}"],
         diagram="gates"),
    beat("symbols", "One head combines signed channel products with positive scales.",
         "For one head, d counts its channels, i selects the receiving query, j the sending key, and c a channel. The gate coordinates g Q and g K are positive scales, while query and key coordinates can have either sign. Calligraphic B is pair bias and M is the validity mask; hold them fixed for the local derivative. Capital S is the scalar logit for a pair, and A is its row-normalized weight.",
         "Positive gate scales do not make signed feature products positive.",
         [r"q_i,k_j,g_i^Q,g_j^K\in\mathbb R^d",
          r"S_{ij}=\frac1{\sqrt d}\sum_{c=1}^dq_{ic}g^Q_{ic}k_{jc}g^K_{jc}+\mathcal B_{ij}+M_{ij}",
          r"g_{ic}^Q,g_{jc}^K>0",
          r"\operatorname{sign}(q_{ic}k_{jc})\in\{-1,0,1\}"],
         diagram="gates", source=SOURCE),
    beat("query_derivative", "A query-gate coordinate directly changes one score row.",
         "Differentiate the logit with respect to query gate coordinate g Q i c, holding other coordinates and paths fixed. Every channel term except channel c is constant with respect to this chosen variable. In the remaining term, remove the differentiated gate factor and keep query, key, key gate, and square-root scaling. Because receiver i stays fixed while j varies, the direct changes occupy row i of the score matrix.",
         "The derivative sign follows the signed query-key channel product.",
         [r"S_{ij}=\frac{q_{ic}g^Q_{ic}k_{jc}g^K_{jc}}{\sqrt d}+\text{constant}",
          r"\frac{\partial g^Q_{ic}}{\partial g^Q_{ic}}=1",
          r"\frac{\partial S_{ij}}{\partial g^Q_{ic}}=\frac{q_{ic}k_{jc}g^K_{jc}}{\sqrt d}",
          r"g_i^Q\longrightarrow S_{i,:}"],
         diagram="matrix", source=SOURCE),
    beat("key_derivative", "A key-gate coordinate directly changes one score column.",
         "Now differentiate with respect to key gate coordinate g K j c instead. The surviving product contains the query coordinate, its query gate, and the selected key coordinate, divided by square root d. Sender j is fixed while receiving queries vary, so these direct changes occupy column j. The derivative can be negative even though the gate itself is positive, because the query and key coordinates can have opposite signs.",
         "Score changes and normalized weight changes differ.",
         [r"S_{ij}=\frac{q_{ic}g^Q_{ic}k_{jc}g^K_{jc}}{\sqrt d}+\text{constant}",
          r"\frac{\partial S_{ij}}{\partial g^K_{jc}}=\frac{q_{ic}g^Q_{ic}k_{jc}}{\sqrt d}",
          r"g_j^K\longrightarrow S_{:,j}",
          r"q_{ic}k_{jc}<0\Longrightarrow\frac{\partial S_{ij}}{\partial g^K_{jc}}<0"],
         diagram="matrix", source=SOURCE),
    beat("softmax_derivative", "Softmax couples every valid weight within the affected row.",
         "Let t index the score being changed, and define delta j t as one when j equals t and zero otherwise. Differentiating row softmax gives A i j times delta j t minus A i t. Increasing one finite score raises its own weight by A times one minus A and lowers competing weights by their product. A column-local score perturbation therefore changes all normalized weights in each affected row, not just that key's coefficient.",
         "Delta is the equality indicator for two key indices.",
         [r"\delta_{jt}=\begin{cases}1&j=t\\0&j\ne t\end{cases}",
          r"\frac{\partial A_{ij}}{\partial S_{it}}=A_{ij}(\delta_{jt}-A_{it})",
          r"\frac{\partial A_{it}}{\partial S_{it}}=A_{it}(1-A_{it})",
          r"j\ne t:\quad\frac{\partial A_{ij}}{\partial S_{it}}=-A_{ij}A_{it}"],
         diagram="matrix", source=SOURCE),
    beat("counterexample", "Amplifying a negative key can reduce its attention.",
         "Choose scalar query one, first key negative one, and second key zero, with head width one and no pair bias. Unit gates give logits negative one and zero, so the first weight is approximately zero point two six eight nine. Double only the first key gate to produce logits negative two and zero. The first weight falls to approximately zero point one one nine two, contradicting the claim that a larger key gate always attracts more attention.",
         "The scale two is the limiting identity-gate endpoint.",
         [r"q=1,\quad k_1=-1,\quad k_2=0,\quad d=1",
          r"S=(-1,0)\Longrightarrow A_1\simeq0.2689",
          r"g_1^K:1\to2\Longrightarrow S=(-2,0)",
          r"A_1\simeq0.1192<0.2689"],
         diagram="attention", weights=[[.26894142,.73105858],[.26894142,.73105858],[.11920292,.88079708],[.11920292,.88079708]],
         source=SOURCE),
    beat("counter_gradient", "The derivative predicts the direction before recomputing softmax.",
         "For the negative first key, the logit derivative with respect to its gate equals negative one. At unit gates, multiply that derivative by A one times one minus A one to get approximately negative zero point one nine six six. This is the local derivative of the first probability with respect to its key scale, not the finite change for a whole unit increase. The negative sign correctly predicts the initial direction, while recomputing softmax gives the exact finite result.",
         "A derivative predicts local slope, not an arbitrary finite step.",
         [r"\frac{\partial S_1}{\partial g_1^K}=qk_1=-1",
          r"\frac{\partial A_1}{\partial g_1^K}=A_1(1-A_1)(-1)",
          r"\left.\frac{\partial A_1}{\partial g_1^K}\right|_{g=1}\simeq-0.1966",
          r"A_1(2)-A_1(1)\simeq-0.1497"],
         diagram="attention", weights=[[.26894142,.73105858]]*4),
    beat("code", "Autograd confirms both signed score and probability derivatives.",
         "This executed example makes the first key gate a differentiable scalar initialized to one. It constructs logits negative gate and zero, then prints the first softmax weight and its automatic derivative. The code separately evaluates the weight at gate two rather than treating the derivative as an exact finite update. Independent assertions compare both results with analytic derivatives and central finite differences.",
         "Autograd and finite differences check the same local slope.",
         [], diagram="none", code="derivative"),
    beat("query_example", "A positive score shrinks when its query coordinate is suppressed.",
         "Return to the two-channel example with query one, zero and perpendicular basis keys. Halving the query's first channel gives logits approximately zero point three five three six and zero, so weights become zero point five eight seven five and zero point four one two five. The second content score remains zero because its nonzero key coordinate still meets a zero query coordinate. This differs from additive bias zero, one, which made the second key win outright.",
         "Coordinate scaling cannot create a missing nonzero product here.",
         [r"g^Q=(0.5,1),\quad g_1^K=g_2^K=(1,1)",
          r"C_{\rm gated}=(0.3536,0)",
          r"A_{\rm gates}\simeq(0.5875,0.4125)",
          r"A_{\rm bias}\simeq(0.4273,0.5727)"],
         diagram="attention", weights=[[.66976155,.33023845],[.587479,.412521],[.587479,.412521],[.4272957,.5727043]]),
    beat("temperature", "Uniform positive query scaling can act like inverse temperature.",
         "Let a be one positive scalar multiplying every channel of a particular query head, while holding the keys and key gates fixed. With no varying additive bias, this multiplies the entire content-score row by a without changing its ordering. Defining temperature T as one over a gives softmax of C divided by T. Larger a sharpens that row's existing preferences, while smaller a flattens them, provided the scores are not already equal.",
         "Temperature requires uniform scaling without varying bias.",
         [r"g_i^Q=a\mathbf1,\quad a>0",
          r"C_{i,:}'=aC_{i,:}",
          r"T=1/a>0",
          r"\operatorname{softmax}(aC)=\operatorname{softmax}(C/T)"],
         diagram="gates", source=SOURCE),
    beat("temperature_bias", "A fixed nonuniform bias breaks the full-temperature equivalence.",
         "Now retain a nonuniform pair bias while scaling only the content row. The actual gated logits are a times C plus B, whereas scaling the original complete logits would give a times C plus a times B. These are generally different because the bias has not been multiplied in the gated computation. Equivalence returns for a constant row bias, which cancels, or if the same scale multiplies the bias too.",
         "Scaling content alone is not scaling content plus relation bias.",
         [r"S_{\rm gated}=aC+\mathcal B",
          r"S_{\rm temperature}=a(C+\mathcal B)=aC+a\mathcal B",
          r"S_{\rm gated}-S_{\rm temperature}=(1-a)\mathcal B",
          r"\mathcal B_j-\mathcal B_k\ne0\Longrightarrow\text{generally different weights}"],
         diagram="bias", source=SOURCE),
    beat("ordering", "Channel-specific gates can reverse which key ranks first.",
         "Take query one, one, first key two, zero, and second key zero, one, with unit key gates. Without query gating, their unscaled dot products are two and one, so the first key wins. Set the query gate to one fifth, one, producing dot products two fifths and one, so the second key wins instead. A positive uniform temperature scale cannot reverse this order, demonstrating why general coordinate gates are more than temperature.",
         "Channel scales change comparison geometry, not only sharpness.",
         [r"q=(1,1),\quad k_1=(2,0),\quad k_2=(0,1)",
          r"q^\top k_1=2>1=q^\top k_2",
          r"g^Q=(0.2,1)\Longrightarrow q'=(0.2,1)",
          r"q'^\top k_1=0.4<1=q'^\top k_2"],
         diagram="attention", weights=[[.66976155,.33023845],[.66976155,.33023845],[.39549685,.60450315],[.39549685,.60450315]]),
    beat("tanh_gradient", "Gate saturation can reduce sensitivity to preactivation changes.",
         "Let z denote one gate preactivation and g equal one plus tanh z. Differentiating gives one minus tanh squared z, which equals one at the identity point z zero. When z has large magnitude, tanh approaches an endpoint and this derivative approaches zero. Positive gates therefore do not guarantee large gradients; the signed score products and the saturation derivative both participate in the chain rule.",
         "The activation derivative multiplies the signed score derivative.",
         [r"g=1+\tanh z",
          r"\frac{dg}{dz}=1-\tanh^2z",
          r"z=0\Longrightarrow dg/dz=1",
          r"\frac{\partial S}{\partial z}=\frac{\partial S}{\partial g}(1-\tanh^2z)"],
         diagram="gates", source=SOURCE),
    beat("loss_path", "Loss gradients depend on the weighted message, not weights alone.",
         "Let L be a scalar loss and let u i be its gradient with respect to output message o i, a d-channel vector. For fixed values and no attention dropout, differentiating the weighted message gives A i j times the difference between value v j and message o i. Taking its dot product with u i yields the loss derivative for score S i j. A large attention weight alone therefore does not measure a large positive contribution to the loss or prediction.",
         "u is the upstream message gradient, not a particle feature.",
         [r"u_i=\frac{\partial L}{\partial o_i}\in\mathbb R^d",
          r"o_i=\sum_jA_{ij}v_j",
          r"\frac{\partial o_i}{\partial S_{ij}}=A_{ij}(v_j-o_i)",
          r"\frac{\partial L}{\partial S_{ij}}=A_{ij}u_i^\top(v_j-o_i)"],
         diagram="flow", source=SOURCE),
    beat("diagonal", "Gating creates a context-dependent diagonal comparison form.",
         "Within one head, collect the channelwise query and key gate products into a diagonal matrix. The gated dot product is the original query transpose times that diagonal matrix times the original key. Its diagonal entries are positive and depend on the event through relation summaries, but queries and keys are different learned projections. This is a useful comparison weighting, not generally a symmetric distance metric with guaranteed triangle inequalities.",
         "Diag places vector entries on a matrix diagonal.",
         [r"g_i^Q\odot g_j^K\in\mathbb R^d",
          r"D_{ij}=\operatorname{diag}(g_i^Q\odot g_j^K)\in\mathbb R^{d\times d}",
          r"q_i'{}^\top k_j'=q_i^\top D_{ij}k_j",
          r"D_{ij}\ \text{depends on both particles}"],
         diagram="gates", source=SOURCE),
    beat("factorization", "Each channel's pairwise gate multipliers form an outer product.",
         "For one fixed channel c, define T i j c as the query gate at receiver i times the key gate at sender j. The complete pair table for that channel is an outer product of two particle vectors, so its rank is at most one. The full content scores sum over d channels and can have rank up to d, just like ordinary content attention. This rank statement applies before softmax, not to the normalized attention matrix.",
         "Rank counts independent directions in a matrix.",
         [r"T_{ijc}=g^Q_{ic}g^K_{jc}",
          r"T_{:,:,c}=g^Q_{:,c}(g^K_{:,c})^\top,\quad\operatorname{rank}(T_{:,:,c})\le1",
          r"S_{\rm content}=\frac{(Q\odot G^Q)(K\odot G^K)^\top}{\sqrt d}",
          r"\operatorname{rank}(S_{\rm content})\le d"],
         diagram="matrix", source=SOURCE),
    beat("compression", "Summarizing a relation row can lose partner-specific information.",
         "A summary reduces an entire relation row to D sub p numbers, so distinct rows can produce the same masked mean. For example, scalar relation rows zero, two and one, one both average to one, despite describing different partners. The same shared gate network must return the same gate when these summaries match, if other inputs are held fixed. Direct pair bias can still distinguish individual relation entries, while the content keys also retain partner-specific information.",
         "Keys retain partner identity despite compressed relation access.",
         [r"r_{i,:}=(0,2),\quad\widetilde r_{i,:}=(1,1)",
          r"s_i=(0+2)/2=(1+1)/2=\widetilde s_i=1",
          r"g(s_i)=g(\widetilde s_i)",
          r"r_{ij}\ne\widetilde r_{ij}\Longrightarrow\mathcal B_{ij}\ \text{can differ}"],
         diagram="gates", source=SOURCE),
    beat("expressivity", "Gates and direct pair bias have different structural capabilities.",
         "Direct pair bias reads each specific relation instead of accessing it only through a particle summary. A learned pair-feature table can have rank exceeding the head width, whereas the gated content term retains its query-key factorization. But bias is not an arbitrary free matrix: shared pair encoders still constrain equal inputs to equal outputs. The reported model uses both mechanisms, and gates-only equivalence or superiority is not established by this structural comparison.",
         "The reported shared model is hybrid, not gates-only.",
         [r"\operatorname{rank}(S_{\rm content})\le d",
          r"\operatorname{rank}(\mathcal B)\ \text{can exceed}\ d",
          r"S=S_{\rm content}+\mathcal B+M",
          r"\text{gates-only}\not\equiv\text{arbitrary pair bias}"],
         diagram="bias", source=SOURCE),
    beat("interpretation", "Average gates do not establish core-to-periphery information flow.",
         "Averaging a gate across channels discards the signed query-key products that actually determine score changes. There is also scale ambiguity: multiplying query gates by a factor and dividing key gates by it preserves their products wherever both transformed gates remain allowed. To demonstrate flow between particle regions, measure actual attention changes grouped by locations, plus weighted value contributions or downstream effects. A correlation between average gates and radius alone does not establish that causal transfer.",
         "Gate averages do not measure message transport.",
         [r"\overline g_i^Q=\frac1d\sum_cg^Q_{ic}",
          r"(ag_i^Q)\odot(g_j^K/a)=g_i^Q\odot g_j^K",
          r"\overline g\ \not\Rightarrow\ \Delta A_{ij}",
          r"\text{test}\ \Delta A_{ij},\quad\Delta(A_{ij}v_j),\quad\Delta o_i"],
         diagram="gates", source=SOURCE),
    beat("exercise1", "Exercise one: calculate signed query-gate score derivatives.",
         "First exercise: choose q equal two, negative one; key k equal three, four; and key gate one half, one. For the first query-gate channel, multiply two by three by one half and divide by square root two, giving three over square root two. For the second, multiply negative one by four by one and divide by square root two, giving negative four over square root two. Increasing the two positive query scales therefore pushes this same pair's score in opposite directions.",
         "The signs depend on coordinates, not whether gates are positive.",
         [r"q=(2,-1),\quad k=(3,4),\quad g^K=(0.5,1),\quad d=2",
          r"\frac{\partial S}{\partial g_1^Q}=\frac{2(3)(0.5)}{\sqrt2}=\frac3{\sqrt2}",
          r"\frac{\partial S}{\partial g_2^Q}=\frac{(-1)(4)(1)}{\sqrt2}=-\frac4{\sqrt2}",
          r"\nabla_{g^Q}S\simeq(2.1213,-2.8284)"],
         diagram="gates"),
    beat("exercise2", "Exercise two: decide whether the temperature analogy applies.",
         "Second exercise: take content scores one and zero, fixed pair bias zero and one, and uniform query scale two. Scaling content alone produces logits two and one, with probabilities approximately zero point seven three one one and zero point two six eight nine. Scaling the original complete logits would instead produce two and two, giving equal probabilities. The answers differ, so a fixed nonuniform bias invalidates the simple full-temperature interpretation.",
         "Compare both complete logit rows before invoking temperature.",
         [r"C=(1,0),\quad\mathcal B=(0,1),\quad a=2",
          r"aC+\mathcal B=(2,1)\Longrightarrow A\simeq(0.7311,0.2689)",
          r"a(C+\mathcal B)=(2,2)\Longrightarrow A=(0.5,0.5)",
          r"(0.7311,0.2689)\ne(0.5,0.5)"],
         diagram="attention", weights=[[.5,.5],[.73105858,.26894142],[.5,.5],[.5,.5]]),
    beat("exercise3", "Exercise three: test what gate means fail to determine.",
         "Third exercise: compare query gates two, zero and zero, two, both with mean one, for query one, one and the two basis keys. The first gate produces content scores square root two and zero, whereas the second produces zero and square root two. Their first-key probabilities are approximately zero point eight zero four four and zero point one nine five six despite identical gate means. These endpoint scales are limiting examples; values arbitrarily close inside the allowed interval give the same conclusion.",
         "Equal averages can hide opposite channel preferences.",
         [r"\overline g^{(a)}=\overline g^{(b)}=1",
          r"g^{(a)}=(2,0)\Longrightarrow C^{(a)}=(\sqrt2,0)",
          r"g^{(b)}=(0,2)\Longrightarrow C^{(b)}=(0,\sqrt2)",
          r"A_1^{(a)}\simeq0.8044,\quad A_1^{(b)}\simeq0.1956"],
         diagram="attention", weights=[[.5,.5],[.80442968,.19557032],[.19557032,.80442968],[.19557032,.80442968]]),
    beat("gds", "The hybrid combines both routes within attention.",
         "Return to our miniature model map: particle embeddings supply blue states, and pair embeddings supply a fixed orange relation tensor. Red gates reshape learned query-key comparisons, while teal bias supplies direct pair-specific logit offsets. Grey masks exclude padded keys before normalization, and yellow values remain the messages being combined. The reported shared configuration and default conditioned configuration differ in their gate summaries, not in this final hybrid attention rule.",
         "Test controlled changes before claiming physical interpretations.",
         [r"R\longrightarrow G^Q,G^K,\mathcal B",
          r"X\longrightarrow Q,K,V",
          r"S=\frac{(Q\odot G^Q)(K\odot G^K)^\top}{\sqrt d}+\mathcal B+M",
          r"O=\operatorname{softmax}(S)V"],
         diagram="gates", source=SOURCE),
    beat("summary", "Signed products and normalization limit simple gate interpretations.",
         "First, query gates directly affect score rows and key gates affect columns, but softmax couples competing weights within each row. Second, larger gates can reduce attention, and temperature equivalence requires restricted uniform scaling without a varying fixed bias. Third, gates compress relations into factorized comparisons while bias retains direct pair access. The next chapter assembles these mechanisms into the complete classifier and traces its actual forward pass.",
         "", [], diagram="none"),
]
SUMMARY = ["Signed products determine how gates change scores.",
           "Softmax couples all competing weights within each row.",
           "Gates and pair bias have complementary structural roles."]
for item in BEATS:
    if item["source"] == "Report: Self-attention, derived step by step":
        item["source"] = SOURCE
CHAPTER = {b["key"]: (b["caption"], b["narration"], b["source"]) for b in BEATS}
CODE = {
    "derivative": """import torch
g = torch.tensor(1., requires_grad=True)
s = torch.stack([-g, g*0])
a = s.softmax(0)[0]
a.backward()
print(a.detach(), g.grad)
print(torch.tensor([-2., 0.]).softmax(0))"""
}


def verify(torch):
    dtype = torch.float64
    g = torch.tensor(1., dtype=dtype, requires_grad=True)
    a = torch.stack([-g, g*0]).softmax(0)[0]
    derivative, = torch.autograd.grad(a, g)
    expected = -a.detach()*(1-a.detach())
    assert torch.allclose(derivative, expected, atol=1e-14)
    def weight(x):
        return torch.tensor([-x, 0.], dtype=dtype).softmax(0)[0]
    finite = (weight(1+1e-5)-weight(1-1e-5))/(2e-5)
    assert torch.allclose(derivative, finite, atol=1e-10)
    assert abs(a.item()-.26894142137) < 1e-10
    assert abs(weight(2).item()-.11920292202) < 1e-10
    q, k = torch.tensor([2., -1.], dtype=dtype), torch.tensor([3., 4.], dtype=dtype)
    gq = torch.tensor([1., 1.], dtype=dtype, requires_grad=True)
    gk = torch.tensor([.5, 1.], dtype=dtype, requires_grad=True)
    score = ((q*gq)*(k*gk)).sum()/2**.5
    dq, dk = torch.autograd.grad(score, (gq, gk))
    assert torch.allclose(dq, q*k*gk.detach()/2**.5)
    assert torch.allclose(dk, q*gq.detach()*k/2**.5)
    assert torch.allclose(dq, torch.tensor([2.1213, -2.8284], dtype=dtype), atol=5e-5, rtol=0)
    content = torch.tensor([1., 0.], dtype=dtype)
    bias = torch.tensor([0., 1.], dtype=dtype)
    assert not torch.allclose((2*content+bias).softmax(0), (2*(content+bias)).softmax(0))
    assert torch.allclose((2*content).softmax(0), (content/.5).softmax(0))
    assert torch.tensor([2., 1.]).argmax() != torch.tensor([.4, 1.]).argmax()
    logits = torch.tensor([.3, -.4], dtype=dtype, requires_grad=True)
    values = torch.tensor([[2., 0.], [0., 4.]], dtype=dtype)
    weights = logits.softmax(0)
    output = weights @ values
    upstream = torch.tensor([.7, -.2], dtype=dtype)
    loss = output @ upstream
    ds, = torch.autograd.grad(loss, logits)
    assert torch.allclose(ds, weights.detach()*((values-output.detach()) @ upstream), atol=1e-14)
    torch.manual_seed(12)
    Q, K, GQ, GK = [torch.randn(5, 2, dtype=dtype) for _ in range(4)]
    GQ, GK = 1+GQ.tanh(), 1+GK.tanh()
    gated = (Q*GQ) @ (K*GK).T
    assert torch.linalg.matrix_rank(gated) <= 2
    assert torch.allclose(gated, (Q*(GQ*1.01)) @ (K*(GK/1.01)).T)
    assert torch.allclose(torch.outer(GQ[:, 0], GK[:, 0]),
                          GQ[:, 0, None]*GK[None, :, 0])
    assert torch.linalg.matrix_rank(torch.outer(GQ[:, 0], GK[:, 0])) == 1
    ga, gb = torch.tensor([2., 0.], dtype=dtype), torch.tensor([0., 2.], dtype=dtype)
    assert ga.mean() == gb.mean()
    aa, ab = (ga/2**.5).softmax(0), (gb/2**.5).softmax(0)
    assert torch.allclose(aa, torch.tensor([.8044, .1956], dtype=dtype), atol=5e-5, rtol=0)
    assert torch.allclose(ab, aa.flip(0))
    z = torch.tensor([0., 3.], dtype=dtype, requires_grad=True)
    dg, = torch.autograd.grad((1+z.tanh()).sum(), z)
    assert torch.allclose(dg, 1-z.detach().tanh()**2)
    assert dg[0] == 1 and dg[1] < .01
    return {"passed": ["negative-key counterexample", "analytic/autograd/finite-difference derivative",
                       "Q and K partial derivatives", "temperature conditions", "channel ordering reversal",
                       "weighted-message loss derivative", "content rank", "per-channel outer-product rank",
                       "scale ambiguity", "equal gate means opposite attention", "tanh identity and saturation"],
            "negative_key_weights": [a.item(), weight(2).item()],
            "probability_derivative": derivative.item(), "query_gradient_exercise": dq.tolist()}
