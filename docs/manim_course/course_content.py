"""Chapters 2-15: original teaching script, animations and executable examples.

Equations are pedagogical expansions of the report; decimal examples are
calculated, not experimental measurements. The running code is authoritative.
"""
from dataclasses import dataclass


@dataclass
class Beat:
    caption: str
    narration: str
    steps: tuple
    kind: str = "derive"
    color: str = "X"
    values: tuple = ()
    source: str = "Technical report, Appendices C-I; local model implementation."


def b(caption, narration, *steps, kind="derive", color="X", values=(), source=None):
    kwargs = {} if source is None else {"source": source}
    return Beat(caption, narration, steps, kind, color, values, **kwargs)


TITLES = {
    2: "Tensors, shapes and masks",
    3: "Logarithms, softmax and cross-entropy",
    4: "Learning from examples",
    5: "Gradients and training",
    6: "Jets and seventeen particle features",
    7: "Four-vectors and pair features",
    8: "Embeddings, LayerNorm and two streams",
    9: "Self-attention from arithmetic",
    10: "Pair bias and attention odds",
    11: "Relational gates",
    12: "Gate effects, gradients and expressivity",
    13: "The complete forward pass",
    14: "Independent PyTorch reconstruction",
    15: "Experiments, workflow and limitations",
}
STEMS = {
    2: "tensors_shapes_masks", 3: "logs_softmax_cross_entropy", 4: "ml_basics",
    5: "gradients_training", 6: "jets_particle_features", 7: "four_vectors_pair_features",
    8: "embeddings_layernorm", 9: "self_attention", 10: "pair_bias",
    11: "relational_gates", 12: "gate_effects_gradients", 13: "complete_forward_pass",
    14: "pytorch_reconstruction", 15: "experiments_limitations",
}

CHAPTERS = {
2: {
"recap": b("Shapes tell us which operations make sense.",
    "Previously, a Hadamard product kept a vector, an inner product returned one number, and an outer product created two axes. Keep that distinction in mind. We now arrange many vectors into batches. The new challenge is not harder multiplication, but remembering what each position means when dimensions are added, exchanged, or reduced.",
    r"a\odot b\in\mathbb R^D", r"a^\top b\in\mathbb R", r"ab^\top\in\mathbb R^{D\times D}"),
"objective": b("You will trace axes and exclude padded particles.",
    "Our goal is to predict the shape of every important tensor before running the model. Capital B counts jets, N counts padded particle slots, and D counts channels. A mask records which slots contain real particles. We will build a small table, reshape it into heads, and calculate a masked average without letting empty slots dilute the answer.",
    r"B=\text{jets},\quad N=\text{slots},\quad D=\text{channels}", r"X\in\mathbb R^{B\times N\times D}", r"m\in\{0,1\}^{B\times N}",kind="flow"),
"address": b("Each axis answers a different address question.",
    "Imagine two jets with three particle slots and four channels. The address X, one, two, three selects one scalar using zero based indexing. First choose a jet, then a particle, then a channel. Selecting a whole row instead gives four numbers. Never interpret a tensor dimension solely from its size: two axes can coincidentally have equal lengths.",
    r"X:(2,3,4)",r"X[1]:(3,4)",r"X[1,2]:(4)",r"X[1,2,3]:()",
    kind="matrix",values=((0,1,2,3),(4,5,6,7),(8,9,10,11))),
"pairs": b("A relation needs both a source and a neighbour.",
    "A particle table needs one particle address, but a pair table needs two. Hold particle i fixed and move across neighbour j. Each cell now stores four physical inputs or, after embedding, sixty four relation channels. The batch address stays outside this grid. Flattening the grid without remembering both particle axes would destroy the meaning of neighbour aggregation.",
    r"U:(B,N,N,4)",r"R:(B,N,N,D_p)",r"R_{b,i,j,:}\in\mathbb R^{D_p}",
    color="R",kind="matrix",values=((11,12,13),(21,22,23),(31,32,33))),
"linear": b("A shared linear layer changes only the channel axis.",
    "Suppose each particle has seventeen features. A linear layer with one hundred twenty eight outputs uses the same matrix on every particle in every jet. In PyTorch the stored weight has output channels first. Multiplication by its transpose converts the final axis while leaving batch and particle axes untouched. Shared weights do not mean identical outputs: particles provide different inputs.",
    r"F:(B,N,17),\quad W:(128,17)",r"X=FW^\top+b",r"X:(B,N,128)",
    kind="flow"),
"broadcast": b("Broadcasting repeats singleton axes without learning weights.",
    "A key mask initially has batch and particle axes. Attention scores have batch, head, query, and key axes. Insert singleton dimensions so the mask aligns with the final key axis. Broadcasting then applies the same valid key decision to every head and every query. It does not create trainable parameters, and it does not automatically mask the query axis.",
    r"m:(B,N)",r"m[:,None,None,:]:(B,1,1,N)",r"S:(B,H,N,N)",color="M"),
"heads": b("Reshaping separates channels before transposing the head axis.",
    "With one hundred twenty eight channels and eight heads, each head receives sixteen channels. First reshape the last dimension into eight groups of sixteen. Then transpose the particle and head axes. Reshape changes how entries are grouped; transpose exchanges axis order. Neither learns a projection. The learned query projection must already have happened before this bookkeeping step.",
    r"d=D/H=128/8=16",r"(B,N,128)\longrightarrow(B,N,8,16)",r"(B,N,8,16)\longrightarrow(B,8,N,16)",
    kind="flow"),
"padding": b("Padding is storage, not an observed zero-valued particle.",
    "Two jets may have different particle counts. We pad the shorter one so both fit a rectangular batch. The boolean mask distinguishes an absent slot from a real particle whose features happen to be zero. Bias terms can make padded embeddings nonzero. Therefore zeroing raw features is not a substitute for masking keys and excluding padded positions from pooling.",
    r"m=(1,1,0)",r"X=(2,6,0)",r"\text{real zero}\ne\text{absent particle}",
    kind="matrix",color="M",values=((2,6,0),)),
"mean": b("The denominator counts real particles, not storage slots.",
    "Calculate the mean of two real scalar features, two and six, followed by one padding slot. The masked numerator is eight. The denominator is two, so the answer is four. Dividing by the storage length would incorrectly produce eight thirds. The same multiplication and reduction are applied separately to every channel of a particle representation.",
    r"\sum_i m_ix_i=1(2)+1(6)+0(0)=8",r"\sum_i m_i=2",r"\bar x=8/2=4",r"8/3\ne4"),
"code": b("The executable reduction preserves the intended axes.",
    "Read the code from top to bottom. A boolean mask is expanded only where a channel dimension is required. Multiplication removes absent entries, and summation removes the particle axis. The result is a single masked average. In the complete model this becomes one vector per jet. Check the denominator separately whenever a masked reduction behaves unexpectedly.",
    "mask_mean",kind="code"),
"exercise1": b("Solve the attention shape before multiplying anything.",
    "Exercise one: there are four jets, thirty two particle slots, one hundred twenty eight channels, and eight heads. How many channels belong to one head, and what is the weight tensor shape? Divide the channel count by eight to obtain sixteen. Each head compares every query with every key, giving four by eight by thirty two by thirty two.",
    r"B=4,\ N=32,\ D=128,\ H=8",r"d=128/8=16",r"A:(4,8,32,32)"),
"exercise2": b("Reject an all-padding jet instead of inventing attention.",
    "Exercise two: what happens when every key is masked? Each score becomes negative infinity. Softmax then has no valid distribution to normalize, and ordinary implementations can return undefined numbers. Clamping a pooling denominator cannot repair this attention problem. The companion rejects empty jets explicitly. A real application must remove empty events or define and test a separate policy.",
    r"m=(0,0,0)",r"S=(-\infty,-\infty,-\infty)",r"\sum_jm_j>0",color="M"),
"summary": b("Preserve axis meanings throughout the calculation.",
    "You can now read particle and relation addresses, split channels into heads, and broadcast a key mask deliberately. Remember the three practical checks: write the axes, count only real particles, and reject an empty jet. Next we turn arbitrary scores into probabilities, so that the row of numbers produced by attention has a precise interpretation.",
    r"X:(B,N,D)",r"R:(B,N,N,D_p)",r"m=1\ \Longleftrightarrow\ \text{real particle}"),
},
3: {
"recap": b("Masks select valid entries before probabilities are formed.",
    "We have learned how tensors store particles and pairs, and why padded slots must not become real neighbours. Attention will soon produce one score for every valid neighbour. A score can be positive, negative, or very large. We need an operation that turns those arbitrary scores into nonnegative weights whose sum is one, while respecting the mask.",
    r"s\in\mathbb R^N",r"a_j\ge0",r"\sum_ja_j=1"),
"objective": b("You will derive probabilities and the classification loss.",
    "This chapter connects exponentials, logarithms, softmax, and cross entropy. We will calculate a three entry softmax exactly, prove that a common score shift changes nothing, and derive the loss for a correct label. These are not four unrelated definitions. Exponentials create positive weights, normalization creates probabilities, and logarithms make their ratios and training penalties easier to analyze.",
    r"s\longrightarrow e^s\longrightarrow p",r"p_y\longrightarrow-\ln p_y",r"\mathcal L\in\mathbb R",kind="flow"),
"logs": b("A logarithm reverses an exponential.",
    "The natural logarithm asks which power of the constant e gives a positive number. Thus exponentiating the logarithm of two gives two again. Products become sums of logarithms, while division becomes subtraction. These identities will explain both logarithmic pair features and attention odds. A logarithm of zero is not finite, so numerical code must handle zero before taking logarithms.",
    r"e^{\ln 2}=2",r"\ln(ab)=\ln a+\ln b",r"\ln(a/b)=\ln a-\ln b",r"a,b>0"),
"softmax": b("Exponentiate first, then divide by the common total.",
    "Let the scores be zero, the logarithm of two, and the logarithm of three. Exponentiation gives one, two, and three. Their sum is six. Dividing every entry by six gives one sixth, two sixths, and three sixths. Notice that the largest score receives the largest weight, but smaller scores still contribute unless they are explicitly masked.",
    r"s=(0,\ln2,\ln3)",r"e^s=(1,2,3)",r"Z=1+2+3=6",r"p=(1/6,2/6,3/6)",
    kind="weights",values=((0,.693147,1.098612),(1,2,3),(1,2,3),(1/6,2/6,3/6))),
"shift": b("A common score shift cancels during normalization.",
    "Add a constant c to every score. Each exponential gains the same factor, e to the c. That factor appears in both numerator and denominator, so it cancels. Adding one hundred therefore leaves the probabilities unchanged. Attention cares about relative scores, not an absolute level. Increasing every bias in one attention row equally cannot redirect that row.",
    r"p_j'=\frac{e^{s_j+c}}{\sum_ke^{s_k+c}}",r"p_j'=\frac{e^ce^{s_j}}{e^c\sum_ke^{s_k}}",r"p_j'=p_j"),
"stability": b("Subtracting the largest score avoids exponential overflow.",
    "The shift identity gives a practical numerical algorithm. Subtract the largest score before exponentiation. The largest shifted score is zero, so its exponential is exactly one, and all others are at most one. This avoids overflow for large positive scores without changing the mathematical answer. PyTorch implements stable routines; do not rebuild unstable probability code inside a training loop.",
    r"c=-\max_js_j",r"s_j-c_{\max}\le0",r"p_j=\frac{\exp(s_j-s_{\max})}{\sum_k\exp(s_k-s_{\max})}"),
"odds": b("Subtracting two logits gives the logarithm of their odds.",
    "Take the ratio of two softmax probabilities. Their common denominator disappears, leaving an exponential of the score difference. Taking the logarithm leaves only that difference. This result is especially useful for pair bias: adding one to one score multiplies its odds relative to unchanged alternatives by e. It does not add one directly to the probability.",
    r"\frac{p_j}{p_k}=\frac{e^{s_j}}{e^{s_k}}=e^{s_j-s_k}",r"\ln(p_j/p_k)=s_j-s_k",r"\Delta s_j=1\Rightarrow\text{odds multiplier}=e"),
"temperature": b("Temperature changes concentration without changing score order.",
    "Divide all logits by a positive temperature. A smaller temperature increases score differences and concentrates probability on the largest scores. A larger temperature moves the distribution toward uniform. Ties remain ties. This is a statement about fixed logits: channelwise query and key gates generally cannot be summarized by one temperature, especially when an additive pair bias is also present.",
    r"p(T)=\operatorname{softmax}(s/T),\quad p(1)=(0.2,0.3,0.5)",
    r"T=0.5\Rightarrow\text{larger score differences}",r"T=2\Rightarrow\text{closer to uniform}",
    kind="weights",values=((.2,.3,.5),(2/19,9/38,25/38),(.262751,.321803,.415446))),
"loss": b("Cross-entropy penalizes low probability for the correct class.",
    "For a single correct class y, the loss is minus the logarithm of its predicted probability. If the correct probability is one half, the loss is about zero point six nine three. If it is one tenth, the loss is about two point three zero three. Both predictions might be wrong by accuracy, but the loss distinguishes how strongly the model disagrees.",
    r"\mathcal L=-\ln p_y",r"p_y=1/2\Rightarrow\mathcal L=\ln2\approx0.6931",r"p_y=1/10\Rightarrow\mathcal L=\ln10\approx2.3026"),
"smoothing": b("Label smoothing assigns some target probability to every class.",
    "The reported training uses label smoothing of zero point one across ten classes. Start with a one hot target, retain ninety percent of it, and add one percent to every class. The correct class receives zero point nine one. Each of the other nine receives zero point zero one. This is not a claim that the observed label is wrong ten percent of the time.",
    r"t=(1-\epsilon)\operatorname{onehot}(y)+\epsilon/C",r"C=10,\ \epsilon=0.1",r"t_y=0.9+0.01=0.91",r"t_{k\ne y}=0.01"),
"code": b("Use raw logits with the cross-entropy function.",
    "The code evaluates the exact softmax example and verifies the shift identity. It also calculates cross entropy from raw logits. Passing already normalized probabilities into the usual cross entropy function applies the wrong transformation, because that function expects scores and internally computes a stable log softmax. For reporting probabilities, use softmax separately after the model returns logits.",
    "softmax",kind="code"),
"exercise1": b("Solve the two-class softmax exactly.",
    "Exercise one: the scores are zero and the logarithm of three. Exponentiation gives one and three, so the total is four. The probabilities are one quarter and three quarters. Now increase both scores by one hundred. The common exponential factor cancels, leaving the same probabilities. This exercise checks the distinction between changing a score difference and changing only its origin.",
    r"e^{(0,\ln3)}=(1,3)",r"p=(1/4,3/4)",r"\operatorname{softmax}(s+100)=p"),
"exercise2": b("A correct prediction can still have substantial loss.",
    "Exercise two: a two class model assigns probability zero point five one to the correct class. Is its accuracy correct, and is its loss zero? The largest probability selects the correct class, so this example counts as correct. But minus the logarithm of zero point five one is about zero point six seven three. Accuracy and loss answer different questions.",
    r"p=(0.51,0.49),\quad y=0",r"\operatorname{argmax}p=0",r"-\ln0.51\approx0.6733"),
"summary": b("Probabilities depend on score differences, not common shifts.",
    "We have derived the probability conversion rather than treating softmax as a magic box. Exponentiation gives positive values, normalization makes their sum one, and cross entropy measures agreement with a target distribution. Keep the stable implementation and the raw logit convention in mind. Next we ask what a model learns from examples and what an experiment can actually demonstrate.",
    r"\operatorname{softmax}(s+c)=\operatorname{softmax}(s)",r"\mathcal L=-\sum_kt_k\ln p_k",r"\text{logits in; scalar loss out}"),
},
4: {
"recap": b("A loss evaluates predictions but does not choose weights.",
    "Softmax converts scores into probabilities, and cross entropy compares those probabilities with a target. So far, every calculation used fixed numbers. Learning introduces adjustable parameters and a dataset of examples. We must separate the act of evaluating a model from changing its weights, and separate good performance on remembered examples from good performance on new jets.",
    r"x\overset{f_\theta}{\longrightarrow}\ell",r"\ell\longrightarrow\mathcal L",r"\theta=\text{trainable parameters}",kind="flow"),
"objective": b("You will separate data, parameters, training, and evaluation.",
    "Our objective is to understand what each part of a machine learning experiment is allowed to see. We will identify features, labels, parameters, and hyperparameters, then trace a small affine classifier. We will also distinguish training, validation, and testing. This vocabulary prevents a synthetic demonstration from being mistaken for a trained physics result, even when both produce ten output scores.",
    r"(x,y)=\text{example}",r"\theta=\text{weights and biases}",r"D,H,L=\text{configuration choices}"),
"features": b("Features describe examples while parameters define the learned rule.",
    "Seventeen measured or derived numbers describe each particle. Those values change from event to event. A weight matrix is shared across events and changes through optimization instead. The embedding width is neither an observation nor an ordinary trainable weight; it is a design choice. Counting input features therefore tells us almost nothing about the total number of model parameters.",
    r"x_i\in\mathbb R^{17}",r"W\in\mathbb R^{128\times17}",r"17\ne17\cdot128+128"),
"affine": b("Every output combines inputs through learned weights and bias.",
    "Consider two input features, two and minus one. The first output uses weights one and three, then adds a bias of one: two minus three plus one is zero. The second uses weights minus two and four, then subtracts one: minus four minus four minus one is minus nine. A matrix product is simply several dot products performed together.",
    r"x=(2,-1),\quad W=\begin{bmatrix}1&3\\-2&4\end{bmatrix}",r"b=(1,-1)",r"h_1=1(2)+3(-1)+1=0",r"h_2=-2(2)+4(-1)-1=-9",
    kind="matrix",values=((1,3),(-2,4))),
"nonlinear": b("Without nonlinearity, stacked affine layers collapse into one.",
    "Apply one affine map and then another without an activation between them. Expanding the expression gives a new matrix product and a new bias, but it is still an affine rule. A nonlinear activation prevents this collapse. The local model uses GELU, which smoothly changes the response rather than clipping every negative input to zero as a hard rectifier would.",
    r"h=W_1x+b_1",r"z=W_2h+b_2",r"z=(W_2W_1)x+(W_2b_1+b_2)",r"z=W_2\operatorname{GELU}(h)+b_2"),
"gelu": b("GELU retains some negative values rather than deleting them.",
    "GELU multiplies its input by the cumulative probability of a standard normal variable below that input. At positive one half, the output is about zero point three four five seven. At negative one half, it is about minus zero point one five four three. The activation is nonlinear and smooth. It does not make an individual neuron a named physical observable.",
    r"\operatorname{GELU}(z)=z\Phi(z)",r"\operatorname{GELU}(0.5)\approx0.3457",r"\operatorname{GELU}(-0.5)\approx-0.1543"),
"dataset": b("Training updates weights using examples from its own split.",
    "One batch contains particle features, pair features, a mask, and one class label per jet. Training computes scores, calculates loss, differentiates, and updates parameters. It repeats across batches. The label belongs to the whole jet, not every particle independently. Copying that label to all particles and training them as separate events would define a different task.",
    r"(F,U,m,y)\longrightarrow\ell=f_\theta(F,U,m)",r"\mathcal L=\operatorname{CE}(\ell,y)",r"\theta\longrightarrow\theta_{\rm next}",kind="flow"),
"splits": b("Validation guides choices while testing measures performance.",
    "Training data drives gradient updates. Validation data helps select settings or checkpoints, but is not used for those gradient updates. The test set is reserved for final evaluation. Repeatedly choosing configurations based on test results leaks information into the experiment. Report which data influenced each decision, not just whether a file was named train, validation, or test.",
    r"\text{train}\longrightarrow\text{parameter updates}",r"\text{validation}\longrightarrow\text{model selection}",r"\text{test}\longrightarrow\text{held-out estimate}",kind="flow"),
"overfit": b("Memorizing a small batch verifies plumbing, not generalization.",
    "A tiny batch overfit test asks whether the model and optimizer can drive training loss down on the same few examples. Failure often reveals a wiring, masking, or loss error. Success does not establish useful physics performance. Our CPU demonstration deliberately memorizes eight synthetic examples. We publish that limitation beside the result rather than presenting its perfect training accuracy as a benchmark.",
    r"\mathcal D_{\rm tiny}\longrightarrow\text{repeated updates}",r"\text{training accuracy}=1\ \not\Rightarrow\ \text{test accuracy}=1",r"\text{synthetic test}\ne\text{JetClass experiment}"),
"dropout": b("Training and evaluation modes deliberately behave differently.",
    "Dropout randomly removes activations during training and rescales surviving ones. With removal probability p, a retained value is divided by one minus p, preserving its conditional expectation. Evaluation disables this randomness. Calling evaluation mode does not turn off automatic differentiation by itself. Use both the appropriate module mode and a no gradient context for efficient deterministic evaluation.",
    r"\operatorname{Drop}(x)=rx/(1-p)",r"r\sim\operatorname{Bernoulli}(1-p)",r"\mathbb E[\operatorname{Drop}(x)\mid x]=x"),
"code": b("The affine calculation matches the runnable tensor operation.",
    "The executable example stores the two rows of the weight matrix and the two bias entries. Matrix multiplication reproduces zero and minus nine. GELU then changes those outputs nonlinearly. Compare this with the expanded hand calculation, not just the output shape. A shape check catches incompatible axes, but it cannot detect every incorrect order of operations.",
    "affine",kind="code"),
"exercise1": b("Count weights and biases separately.",
    "Exercise one: how many trainable parameters are in a linear layer from seventeen inputs to one hundred twenty eight outputs? Each output needs seventeen weights, giving two thousand one hundred seventy six weights. Each output also has one bias, adding one hundred twenty eight. The total is two thousand three hundred four. The number of particles does not appear because weights are shared.",
    r"17\cdot128=2176",r"2176+128=2304",r"\text{parameter count independent of }N"),
"exercise2": b("Evaluation must not update parameters from test labels.",
    "Exercise two: suppose test loss is computed and backward is called, but no optimizer step follows. Have the weights changed? Not from backward alone: it accumulates gradients. However, this is unnecessary and risks accidental later use of test gradients. Correct evaluation uses no gradient tracking and never uses test labels to choose a parameter update. The clean workflow removes that ambiguity.",
    r"\operatorname{backward}\longrightarrow\nabla_\theta\mathcal L",r"\operatorname{step}\longrightarrow\theta_{\rm next}",r"\text{test: no gradient updates}"),
"summary": b("Implementation checks are not scientific evidence.",
    "You now know which numbers are observations, which are learned, and which are configuration choices. You can trace an affine layer and explain why nonlinearity matters. You can also distinguish memorization, validation selection, and held-out testing. Next we derive how a loss produces the parameter changes that learning requires, instead of treating the optimizer as an unexplained instruction.",
    r"\text{features}\ne\text{parameters}",r"\text{training}\ne\text{evaluation}",r"\text{overfit check}\ne\text{generalization evidence}"),
},
5: {
"recap": b("Learning changes parameters to reduce a chosen objective.",
    "We have built a small network from affine maps and nonlinearities, and separated training from evaluation. Now we need a direction for changing its parameters. Guessing random changes would waste information already present in the calculation. A derivative tells us how a small change affects an output, and a gradient collects that information for all trainable parameters.",
    r"\theta\longrightarrow\mathcal L(\theta)",r"\nabla_\theta\mathcal L",r"\theta_{\rm next}=\theta-\eta\nabla_\theta\mathcal L"),
"objective": b("You will derive and execute one complete learning update.",
    "We will start with a scalar prediction so every multiplication remains visible. Then we will connect the chain rule to a computation graph, extend it to an affine layer, and explain the actual training loop. The goal is not to derive every optimizer theorem. It is to know what backward computes, what step changes, and why gradients must be cleared deliberately.",
    r"x\longrightarrow\widehat y\longrightarrow\mathcal L",r"\mathcal L\longrightarrow\nabla_w\mathcal L",r"w\longrightarrow w_{\rm next}",kind="flow"),
"derivative": b("A derivative measures local sensitivity, not a finite prediction.",
    "For the function x squared, compare the output at x plus a small change with the output at x. Expand the square and divide by the change. The result is twice x plus the change, approaching twice x as the change shrinks. At x equal to three, the derivative is six. This slope approximates sufficiently small changes, not arbitrary jumps.",
    r"\frac{(x+\delta)^2-x^2}{\delta}",r"\frac{2x\delta+\delta^2}{\delta}=2x+\delta",r"\delta\to0\Rightarrow f'(x)=2x"),
"forward": b("Start with a prediction and an explicitly defined loss.",
    "Take input two, target five, weight one, and bias zero. The affine prediction is two. Define the loss as half the squared error, so the error is minus three and the loss is four point five. The factor one half cancels the factor two when differentiating the square. This convenient example is regression, not the classification loss used for jets.",
    r"x=2,\ y=5,\ w=1,\ b=0",r"\widehat y=wx+b=2",r"e=\widehat y-y=-3",r"\mathcal L=e^2/2=4.5"),
"chain": b("Multiply local derivatives along the computation path.",
    "The loss changes with prediction at a rate equal to the error. The prediction changes with the weight at a rate equal to the input. Multiplying gives the weight gradient, minus three times two, which is minus six. The bias gradient is minus three because prediction changes one for one with bias. Negative gradients mean increasing these parameters locally reduces this loss.",
    r"\frac{\partial\mathcal L}{\partial\widehat y}=\widehat y-y=-3",r"\frac{\partial\widehat y}{\partial w}=x=2",r"\frac{\partial\mathcal L}{\partial w}=(-3)(2)=-6",r"\frac{\partial\mathcal L}{\partial b}=-3"),
"update": b("Subtracting a negative gradient increases the parameter.",
    "Use learning rate zero point one. Subtract zero point one times minus six from the weight, giving one point six. The new bias is zero point three. The updated prediction is three point five, closer to the target five. Its loss is one point one two five. This single step decreases the example loss, but an arbitrarily large learning rate need not do so.",
    r"w'=1-0.1(-6)=1.6",r"b'=0-0.1(-3)=0.3",r"\widehat y'=1.6(2)+0.3=3.5",r"\mathcal L'=(-1.5)^2/2=1.125"),
"affine_grad": b("An affine weight gradient is an outer product.",
    "For a vector output h equal to W x plus b, let delta denote the loss gradient arriving at h. Each weight connects one input to one output. Its gradient is the corresponding output gradient times that input value. Collecting all such products gives delta times x transpose. The input gradient uses W transpose, which sends the signal backward through the same connections.",
    r"h=Wx+b,\quad\delta=\nabla_h\mathcal L",r"\nabla_W\mathcal L=\delta x^\top",r"\nabla_b\mathcal L=\delta",r"\nabla_x\mathcal L=W^\top\delta"),
"classification": b("Cross-entropy sends predicted probability minus target backward.",
    "For softmax followed by cross entropy, the derivative with respect to each logit simplifies to predicted probability minus target probability. An underpredicted correct class has a negative gradient, encouraging its logit upward under gradient descent. The gradients across classes sum to zero. This reflects the fact that a common logit shift leaves probabilities and loss unchanged.",
    r"\frac{\partial\mathcal L}{\partial\ell_k}=p_k-t_k",r"p=(0.2,0.8),\quad t=(1,0)",r"\nabla_\ell\mathcal L=(-0.8,0.8)",r"\sum_k(p_k-t_k)=0"),
"residual": b("A residual connection adds a direct gradient path.",
    "A residual block returns its input plus a learned update. Differentiating gives the identity matrix plus the update's Jacobian. The identity term allows a gradient contribution to pass directly backward without multiplying only through the learned branch. This helps deep optimization, but it is not a guarantee against vanishing or exploding gradients. The learned branch and repeated composition still matter.",
    r"y=x+f(x)",r"\frac{\partial y}{\partial x}=I+J_f(x)",r"\nabla_x\mathcal L=\nabla_y\mathcal L+J_f^\top\nabla_y\mathcal L"),
"loop": b("Clear gradients, differentiate, then update in that order.",
    "PyTorch accumulates gradients unless they are cleared. A basic step therefore begins by clearing them, computes a new forward pass and loss, calls backward, and then calls the optimizer. Gradient clipping limits the update signal's norm before the step. The reported training uses AdamW and a learning rate schedule; our scalar derivation explains gradient descent, not every internal AdamW operation.",
    r"\text{zero gradients}\to\text{forward}\to\text{loss}",r"\text{backward}\to\text{clip}\to\text{step}",r"\text{new batch}\to\text{repeat}",kind="flow"),
"code": b("Automatic differentiation reproduces the hand-derived gradients.",
    "The displayed code enables gradients for the weight and bias, constructs the same loss, and calls backward. The printed gradients are minus six and minus three. The parameter update is performed without recording a new gradient graph. That separation prevents the optimizer's bookkeeping from becoming part of the next model calculation. The companion also checks finite gradients throughout the complete architecture.",
    "gradient",kind="code"),
"exercise1": b("Solve the bias-only update with the same error.",
    "Exercise one: keep weight fixed at one and update only the bias using learning rate zero point one. The prediction begins at two, so the bias gradient is still minus three. The new bias is zero point three and the new prediction is two point three. It improves less than the joint update because the weight was deliberately held fixed.",
    r"\partial_b\mathcal L=-3",r"b'=0.3,\quad w'=1",r"\widehat y'=1(2)+0.3=2.3"),
"exercise2": b("The first backward pass need not move every gate weight.",
    "Exercise two: an identity gate has a zero initialized final linear weight. Can its earlier linear layer receive a nonzero gradient through that final weight on the first backward pass? The chain rule multiplies by the zero weight, so that path initially gives zero. The final layer can learn first. Zero gradients here are intentional initialization behavior, not automatically broken automatic differentiation.",
    r"a=W_2h+b_2,\quad W_2=0",r"\nabla_h\mathcal L=W_2^\top\nabla_a\mathcal L=0",r"\nabla_{W_2}\mathcal L=(\nabla_a\mathcal L)h^\top"),
"summary": b("Gradients connect the objective to every learned component.",
    "We derived a complete update and verified it with automatic differentiation. The same chain rule connects classification loss to particle projections, pair embeddings, biases, and gates. Remember the distinction between computing gradients and applying them. Next we return to the data: before learning any representation, we must understand what the seventeen particle features and the four pair inputs actually describe.",
    r"\text{forward computes values}",r"\text{backward computes sensitivities}",r"\text{optimizer changes parameters}"),
},
6: {
"recap": b("Training adjusts a rule applied to physical observations.",
    "We now understand both the forward calculation and the gradient update. But a mathematically correct network can still be scientifically meaningless if its inputs are misunderstood. A jet is a collection of reconstructed particles, and its label describes the whole collection. This chapter connects the arrays to observations before the next chapter derives relationships between pairs.",
    r"\{f_i\}_{i=1}^{N_{\rm real}}\longrightarrow\text{one jet}",r"f_i\in\mathbb R^{17}",r"y\in\{0,\ldots,9\}",kind="flow"),
"objective": b("You will identify all seventeen particle inputs.",
    "We will group the seventeen inputs by purpose rather than memorize an unexplained list. Some describe momentum and energy, some locate particles relative to the jet, some encode particle identity, and some describe displaced tracks. We will also follow sorting, truncation, and padding in the cache. These preprocessing choices define the model's actual information and must be reproduced at inference.",
    r"4+3+6+4=17",r"\text{energy; direction; identity; tracks}",r"F:(B,N,17)"),
"momentum": b("Transverse momentum measures motion perpendicular to the beam.",
    "The beam direction defines the longitudinal axis. The two perpendicular momentum components combine into transverse momentum by the Pythagorean rule. Energy is a different quantity, even though both are related for relativistic particles. The model receives logarithms of particle transverse momentum and energy, as well as logarithms of their fractions relative to the jet. Logs compress a wide numerical range.",
    r"p_T=\sqrt{p_x^2+p_y^2}",r"\ln p_T,\quad\ln E",r"\ln(p_T/p_{T,\rm jet}),\quad\ln(E/E_{\rm jet})"),
"scale": b("The cache applies fixed numerical transforms before embedding.",
    "The first four features are not raw logarithms. The cache shifts and scales them using fixed constants. For example, the transverse momentum feature is zero point seven times the logarithm of transverse momentum minus one point seven. These constants are preprocessing conventions, not trainable model weights. Changing units without adjusting this convention changes the feature values supplied to the network.",
    r"f_0=0.7(\ln p_T-1.7)",r"f_1=0.7(\ln E-2.0)",r"f_2=0.7(\ln(p_T/p_{T,\rm jet})+4.7)",r"f_3=0.7(\ln(E/E_{\rm jet})+4.7)",
    source="data\\jetclass_cache.py, _process_chunk; report feature table."),
"direction": b("Relative direction is measured with respect to the jet.",
    "The next three features describe where a particle sits relative to the jet axis. They include relative pseudorapidity, wrapped relative azimuth, and angular distance. Azimuth is periodic: crossing from just below pi to just above minus pi should not create a separation near two pi. Pseudorapidity and true rapidity are different definitions, even when they become close for nearly massless particles.",
    r"f_4=\eta-\eta_{\rm jet}",r"f_5=\operatorname{wrap}(\phi-\phi_{\rm jet})",r"f_6=\sqrt{(\Delta\eta)^2+(\Delta\phi)^2}"),
"identity": b("Charge and identity indicators add non-kinematic information.",
    "The next six channels include electric charge and five identity indicators. Those indicators describe electron, muon, photon, charged hadron, and neutral hadron categories in the dataset. They are inputs, not the ten output jet classes. Confusing the two would turn a constituent descriptor into a target label. The relation stream does not directly consume these identity channels in this implementation.",
    r"f_7=q",r"(f_8,\ldots,f_{12})=(e,\mu,\gamma,h^\pm,h^0)",r"\text{particle identity}\ne\text{jet class}"),
"tracks": b("Track displacement features can carry lifetime-related information.",
    "The last four features describe transverse and longitudinal impact parameters and their uncertainties. The cache applies hyperbolic tangent to the displacement values and clips uncertainty values into the interval from zero to one. The numerical convention belongs to the dataset. A large transformed value does not directly establish a specific particle origin; the classifier learns statistical associations from labeled jets.",
    r"f_{13}=\tanh d_0,\quad f_{14}=\operatorname{clip}(\sigma_{d_0},0,1)",r"f_{15}=\tanh d_z,\quad f_{16}=\operatorname{clip}(\sigma_{d_z},0,1)",r"13,14,15,16\longrightarrow4\text{ channels}"),
"cache": b("Sorting and truncation change which constituents are retained.",
    "The cache sorts constituents by transverse momentum and retains at most the configured particle limit, normally one hundred twenty eight. This is a selection rule before the transformer, not a learned attention decision. Batches are padded only to their longest retained jet, reducing quadratic pair storage. The cache stores features compactly, then supplies floating point tensors for the model.",
    r"\text{particles}\to\text{sort by }p_T\to\text{retain first }128",r"N_{\rm batch}=\max_bN_b",r"U:(B,N_{\rm batch},N_{\rm batch},4)",kind="flow"),
"symmetry": b("Sorting does not replace a permutation-invariance check.",
    "The model has no position embedding tied to the stored constituent order. Reordering particles, both pair axes, and the mask consistently should leave the jet logits unchanged in deterministic evaluation. Sorting is useful for truncation and batching, but the network should not depend on the accidental order of retained particles. We test this directly rather than assuming it from a diagram.",
    r"F'=PF,\quad U'=PUP^\top,\quad m'=Pm",r"f(F',U',m')=f(F,U,m)",r"\text{evaluation mode; consistent permutation}"),
"code": b("The code evaluates the fixed logarithmic feature transform.",
    "Here we use the illustrative transverse momenta eighty and thirty from the report's pair example. The printed feature values come from the cache's formula, not from a trained embedding. This distinction matters: preprocessing produces deterministic input features, while learned embeddings later mix them. A reproducible inference pipeline must use the same feature order and transforms as training.",
    "features",kind="code"),
"exercise1": b("Count the groups rather than guessing the feature width.",
    "Exercise one: count the input channels from their groups. Four energy and fraction features, three directional features, six charge and identity features, and four track features sum to seventeen. None of these groups adds a particle axis. They all occupy the last channel axis of the same particle row. The mask remains a separate boolean array.",
    r"4+3=7",r"7+6=13",r"13+4=17",r"F:(B,N,17)"),
"exercise2": b("Pair features do not directly encode every particle feature.",
    "Exercise two: does the relation stream directly receive the electron indicator or transverse impact parameter? No. Its four inputs are derived from pairs of four-vectors, as we will calculate next. Particle states do receive the seventeen features. Conditioned gates can later depend on those states, but that is a different computational path from placing identity or track values directly in the pair input.",
    r"F_i\in\mathbb R^{17}",r"U_{ij}=\operatorname{physics}(v_i,v_j)\in\mathbb R^4",r"\text{conditioning through }X\ne\text{direct input in }U"),
"summary": b("The input pipeline is part of the model specification.",
    "We have identified all seventeen channels and separated constituent identity from the jet label. Fixed transforms, sorting, truncation, and masking determine what observations reach the network. The next step is to compute four pair features from four-vectors. Keep the unit convention and the difference between pseudorapidity and rapidity visible, because neither should disappear behind an embedding layer.",
    r"F:\text{seventeen particle features}",r"U:\text{four pair features}",r"m:\text{real-particle indicator}"),
},
7: {
"recap": b("Particle observations and pair relationships carry different information.",
    "The particle table contains seventeen channels, including identity and track information. The relation table instead starts from four-vectors for pairs of constituents. We will derive its four channels rather than labeling an arrow as physics. Our example uses two transverse momenta and small angular separations from the report, with massless constituents for the illustrative invariant-mass calculation.",
    r"v_i=(p_x,p_y,p_z,E)_i",r"U_{ij}\in\mathbb R^4",r"U\longrightarrow R",color="R",kind="flow"),
"objective": b("You will compute every pair feature numerically.",
    "We will derive angular distance, transverse momentum scale, momentum sharing, and pair invariant mass, then take their logarithms in the implementation's order. We will also distinguish a self-pair from a pair involving padding. At the end, a runnable calculation will reproduce all four reported illustrative values and the raw zero convention for the diagonal.",
    r"U_{ij}=(\ln\Delta,\ln k_T,\ln z,\ln m^2)",r"\Delta,\ k_T,\ z,\ m^2>0\text{ after clamping}",r"U:(B,N,N,4)",color="R"),
"fourvector": b("Energy and momentum combine with a minus-sign convention.",
    "A four-vector here is stored as momentum x, momentum y, momentum z, then energy. Its invariant mass squared is energy squared minus the squared spatial momentum magnitude. To form a pair, add the four-vectors first and then compute that invariant. Adding individual masses is not the same operation. The ordering is a code convention that must be checked at the interface.",
    r"v=(p_x,p_y,p_z,E)",r"m^2=E^2-p_x^2-p_y^2-p_z^2",r"v_{ij}=v_i+v_j",color="R"),
"rapidity": b("The pair distance uses rapidity and wrapped azimuth.",
    "Rapidity is half the logarithm of energy plus longitudinal momentum divided by energy minus longitudinal momentum. Azimuth comes from the two transverse components. Subtract the two rapidities, and wrap the azimuth difference into the interval from minus pi to pi. The cache clamps logarithm arguments for numerical safety. This pair definition uses rapidity, not the pseudorapidity feature in the particle table.",
    r"y=\tfrac12\ln\frac{E+p_z}{E-p_z}",r"\phi=\operatorname{atan2}(p_y,p_x)",r"\Delta\phi=\operatorname{wrap}(\phi_i-\phi_j)",color="R"),
"distance": b("The angular distance follows a two-dimensional Pythagorean rule.",
    "Use rapidity difference zero point one five and azimuth difference minus zero point one five. Square both differences, add them, and take the square root. The result is about zero point two one two one three. The sign of the azimuth difference disappears in this distance, but wrapping before squaring remains essential for particles near the angular boundary.",
    r"\Delta y=0.15,\quad\Delta\phi=-0.15",r"\Delta=\sqrt{0.15^2+(-0.15)^2}",r"\Delta=\sqrt{0.045}\approx0.21213",color="R"),
"kt": b("The softer transverse momentum sets the pair momentum scale.",
    "The transverse momenta are eighty and thirty in the report's numerical units. The smaller is thirty. Multiply it by the angular distance to obtain the relative transverse momentum scale, approximately six point three six four. A close pair and a wide pair with the same constituent momenta therefore produce different values. This quantity is not the jet's total transverse momentum.",
    r"\min(p_{T,i},p_{T,j})=30",r"k_T=\min(p_{T,i},p_{T,j})\Delta",r"k_T=30(0.21213)\approx6.364",color="R"),
"sharing": b("Momentum sharing compares the softer constituent with their sum.",
    "Divide the smaller transverse momentum by the sum of both transverse momenta. Thirty divided by one hundred ten is approximately zero point two seven two seven three. This sharing fraction is at most one half because the numerator is the smaller contribution. Equal transverse momenta give one half, while a very soft second constituent pushes the fraction toward zero.",
    r"z=\frac{\min(p_{T,i},p_{T,j})}{p_{T,i}+p_{T,j}}",r"z=\frac{30}{80+30}=\frac3{11}",r"z\approx0.27273\le1/2",color="R"),
"mass": b("The massless pair formula still depends on both separations.",
    "For this illustrative massless pair, invariant mass squared equals twice the transverse momentum product times hyperbolic cosine of rapidity difference minus cosine of azimuth difference. Substitution gives approximately one hundred eight point zero zero zero two. The production cache instead adds the supplied four-vectors and evaluates the invariant directly, which also handles nonzero constituent masses.",
    r"m^2=2p_{T,i}p_{T,j}[\cosh(\Delta y)-\cos(\Delta\phi)]",r"m^2=4800[\cosh(0.15)-\cos(-0.15)]",r"m^2\approx108.0002",color="R"),
"logs": b("Take the four logarithms in the documented channel order.",
    "We now have all four positive quantities. Their natural logarithms are approximately minus one point five five zero five, one point eight five zero seven, minus one point two nine nine three, and four point six eight two one. The last channel is the logarithm of mass squared, not the logarithm of mass. Exchanging these conventions would change the network input.",
    r"(\Delta,k_T,z,m^2)\approx(0.21213,6.364,0.27273,108.0002)",r"U_{ij}=(\ln\Delta,\ln k_T,\ln z,\ln m^2)",r"U_{ij}\approx(-1.5505,1.8507,-1.2993,4.6821)",color="R"),
"zeros": b("Raw zero relations can become nonzero embedded relations.",
    "The cache sets self-pairs and pairs involving padding to a raw zero vector after forming safe logarithms. But an affine embedding includes biases, so embedding a zero vector need not produce zero. Real self-pairs can still participate in aggregation and attention. Padded keys are separately masked. Do not silently remove all diagonal relations under the mistaken belief that a zero input means no computation.",
    r"U_{ii}=0",r"R_{ii}=W_2\operatorname{GELU}(b_1)+b_2",r"R_{ii}\text{ need not equal }0",color="R"),
"code": b("The four-vector implementation reproduces the illustrative numbers.",
    "The executable example constructs massless four-vectors from transverse momentum, rapidity, and azimuth. It calls the independent pair-feature function and prints one off-diagonal pair. Tests compare that function against the repository cache implementation, including symmetry and diagonal zeros. Agreement in this small example is a useful check, not a proof that every possible detector input has been validated.",
    "pair",kind="code",color="R"),
"exercise1": b("Equal transverse momenta give the maximum sharing fraction.",
    "Exercise one: both constituents have transverse momentum thirty. What is the sharing fraction? The smaller value is thirty, and the sum is sixty, so the answer is one half. No angular information enters this fraction. Two pairs can have the same sharing fraction but different distances, momentum scales, and masses. That is why the four channels are not interchangeable.",
    r"p_{T,i}=p_{T,j}=30",r"z=30/(30+30)=1/2",r"z\text{ does not determine }\Delta"),
"exercise2": b("A constituent pair need not reconstruct the parent mass.",
    "Exercise two: should every pair inside a heavy-particle jet peak at the full parent mass? No. A particular pair usually contains only part of the decay products and radiation. The invariant of all relevant constituents and the invariant of an arbitrary pair are different observables. Pair features can be useful without assigning each pair a direct parent-mass interpretation.",
    r"v_{\rm pair}=v_i+v_j",r"v_{\rm jet}=\sum_kv_k",r"v_{\rm pair}^2\ne v_{\rm jet}^2\text{ in general}"),
"summary": b("The relation input is a calculated table, not a label.",
    "We derived angular distance, momentum scale, sharing fraction, and pair mass, then reproduced their logarithmic channels. We also separated raw diagonal zeros from learned embedded values and kept unit conventions explicit. Next we turn these inputs into learned representations, with a complete LayerNorm calculation before discussing how particle states and relations enter different computational paths.",
    r"v_i,v_j\longrightarrow U_{ij}\longrightarrow R_{ij}",r"U_{ii}=0\not\Rightarrow R_{ii}=0",r"\text{pair mass}\ne\text{parent mass in general}",color="R"),
},
8: {
"recap": b("The model receives seventeen particle channels and four pair channels.",
    "We have followed the input pipeline all the way from observations to particle and pair tensors. The four pair channels are deterministic physical calculations; they are not yet the learned relation representation. Embedding layers now change channel widths and mix inputs. Before tracing those layers, we must understand normalization, including the learned scaling that follows its statistical calculation.",
    r"F:(B,N,17)",r"U:(B,N,N,4)",r"F,U\longrightarrow X,R",kind="flow"),
"objective": b("You will calculate LayerNorm and distinguish the embedding paths.",
    "This chapter derives LayerNorm using the four numbers two, four, six, and eight. We will calculate the mean, variance, standardized values, and learned affine transform. Then we will identify exactly where normalization and dropout appear in the particle and pair embeddings. The two streams differ in both shape and module structure; they are not merely two colors for queries and keys.",
    r"x\to\mu,v\to\widehat x\to\gamma\odot\widehat x+\beta",r"X:(B,N,128)",r"R:(B,N,N,64)",kind="flow"),
"mean": b("LayerNorm computes statistics within one particle's channels.",
    "Take the vector two, four, six, eight. Add the entries to get twenty, then divide by the channel count four to get mean five. Subtract this mean from every channel. The centered vector is minus three, minus one, one, three. This calculation happens independently for each particle; it does not average different jets together as a batch statistic.",
    r"x=(2,4,6,8)",r"\mu=(2+4+6+8)/4=5",r"x-\mu=(-3,-1,1,3)",
    kind="matrix",values=((2,4,6,8),(-3,-1,1,3))),
"variance": b("Use the channel count rather than the unbiased sample denominator.",
    "Square the centered entries to get nine, one, one, nine. Their sum is twenty. Divide by four, not three, to obtain variance five. PyTorch LayerNorm uses this population-style denominator. The standard deviation used in normalization is the square root of variance plus epsilon, where the default epsilon is ten to the minus five. Epsilon prevents division by zero for constant vectors.",
    r"(x-\mu)^2=(9,1,1,9)",r"v=(9+1+1+9)/4=5",r"\sigma=\sqrt{5+10^{-5}}"),
"standardize": b("Divide each centered channel by the same positive scale.",
    "Divide minus three, minus one, one, three by the square root of five plus epsilon. The standardized vector is approximately minus one point three four two, minus zero point four four seven, zero point four four seven, and one point three four two. It still has four entries. Its mean is zero, and its variance is slightly below one because epsilon is included.",
    r"\widehat x_c=\frac{x_c-\mu}{\sqrt{v+\epsilon}}",r"\widehat x\approx(-1.342,-0.447,0.447,1.342)",r"\operatorname{Var}(\widehat x)=\frac{v}{v+\epsilon}"),
"affine": b("Learned scaling is elementwise and preserves all four channels.",
    "Use gamma equal to one, two, one half, one, and beta equal to zero point one in every channel. Multiply corresponding entries rather than taking a dot product, then add beta. The result is approximately minus one point two four two, minus zero point seven nine four, zero point three two four, and one point four four two. Learned affine scaling need not preserve zero mean.",
    r"\gamma=(1,2,0.5,1),\quad\beta=(0.1,0.1,0.1,0.1)",r"\operatorname{LN}(x)=\gamma\odot\widehat x+\beta",r"\operatorname{LN}(x)\approx(-1.242,-0.794,0.324,1.442)"),
"particle": b("Particle embedding normalizes before its two affine transforms.",
    "Read the particle path in execution order: normalize seventeen channels, project to one hundred twenty eight, apply GELU, project again at the same width, then apply dropout. The initial LayerNorm has learned scale and bias for seventeen channels. A seventeen-channel input vector therefore becomes a one hundred twenty eight-channel learned particle state without changing the particle or batch axes.",
    r"F\to\operatorname{LN}_{17}\to\operatorname{Linear}_{17,128}",r"\to\operatorname{GELU}\to\operatorname{Linear}_{128,128}",r"\to\operatorname{Dropout}\to X",kind="flow"),
"pair": b("Pair embedding has neither LayerNorm nor dropout.",
    "The pair path projects four channels to sixty four, applies GELU, and projects sixty four channels to sixty four. There is no LayerNorm and no dropout inside this module in the local code. The resulting relation tensor retains both particle axes. It is computed once per forward pass, then reused by every particle block rather than recurrently rewritten.",
    r"U\to\operatorname{Linear}_{4,64}\to\operatorname{GELU}",r"\to\operatorname{Linear}_{64,64}\to R",r"R:(B,N,N,64)",kind="flow",color="R"),
"counts": b("Count trainable storage separately from activation storage.",
    "The particle embedding has thirty four normalization parameters, two thousand three hundred four parameters in its first affine layer, and sixteen thousand five hundred twelve in its second. The total is eighteen thousand eight hundred fifty. Pair embedding has four thousand four hundred eighty. These counts do not grow with particle multiplicity, but the relation activation tensor grows quadratically with that multiplicity.",
    r"2(17)+(17\cdot128+128)+(128^2+128)=18850",r"(4\cdot64+64)+(64^2+64)=4480",r"\operatorname{size}(R)=BN^2D_p"),
"code": b("The executable LayerNorm matches every hand-calculated stage.",
    "The code computes mean and variance directly, then applies the learned channel scales and shifts. It also calls the framework LayerNorm operation with the same parameters. Their outputs agree within floating point tolerance. This is stronger than checking only that four values emerge: a dot product followed by broadcasting might have the right final shape while implementing the wrong operation.",
    "layernorm",kind="code"),
"exercise1": b("A constant vector becomes zero before learned shifting.",
    "Exercise one: normalize the vector one, one, one, with gamma one and beta zero. Its mean is one and its variance is zero. Every centered entry is zero, and division by the square root of epsilon leaves zero. There is no division by zero. If beta were nonzero, the final output would equal beta, illustrating the role of the learned shift.",
    r"x=(1,1,1),\quad\mu=1,\quad v=0",r"\widehat x=(0,0,0)/\sqrt\epsilon=(0,0,0)",r"\gamma=1,\ \beta=0\Rightarrow\operatorname{LN}(x)=0"),
"exercise2": b("Dual stream does not mean query stream and key stream.",
    "Exercise two: identify the two streams by their addresses. Particle states need batch, particle, and channel addresses. Relations need batch, first particle, second particle, and relation channel addresses. Queries and keys are projections formed from particle states inside attention. They are not the two streams named by this architecture. The relation table remains fixed during the particle-block recurrence.",
    r"X:(B,N,D),\quad R:(B,N,N,D_p)",r"Q=XW_Q^\top+b_Q,\quad K=XW_K^\top+b_K",r"X^{(\ell+1)}\text{ changes; }R\text{ is reused}"),
"summary": b("Embeddings create different representations for particles and relations.",
    "LayerNorm is now a fully specified calculation rather than a label. We know its statistics, epsilon, learned scale, and learned shift. We also know the exact two embedding paths and their parameter counts. Next we derive self attention with two keys and two values, so that every weight and every output channel can be checked by hand.",
    r"\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{v+\epsilon}}+\beta",r"F\to X,\quad U\to R",r"\text{fixed }R\ne\text{recurrent second transformer}"),
},
}

CODE = {
"mask_mean": """import torch
x = torch.tensor([2., 6., 0.])
m = torch.tensor([1, 1, 0])
print((x*m).sum()/m.sum())
print(m[None,None,:].shape)
""",
"softmax": """import torch
s = torch.tensor([1.,2.,3.]).log()
print(s.softmax(0).round(decimals=4))
print(torch.allclose(
    s.softmax(0), (s+100).softmax(0)))
print(torch.nn.functional.cross_entropy(
    s[None], torch.tensor([2])).round(decimals=4))
""",
"affine": """import torch
x = torch.tensor([2., -1.])
w = torch.tensor([[1.,3.],[-2.,4.]])
b = torch.tensor([1., -1.])
print(w @ x + b)
print(torch.nn.functional.gelu(w @ x + b))
""",
"gradient": """import torch
w = torch.tensor(1., requires_grad=True)
b = torch.tensor(0., requires_grad=True)
loss = ((2*w+b)-5).square()/2
loss.backward()
print(w.grad, b.grad)
with torch.no_grad():
    w -= .1*w.grad; b -= .1*b.grad
print(2*w+b)
""",
"features": """import torch
pt = torch.tensor([80., 30.])
f0 = .7*(pt.log()-1.7)
print(f0.round(decimals=4))
""",
"pair": """import torch
from gds_part_reconstruction import pair_features
pt = torch.tensor([80.,30.])
y = torch.tensor([.15,0.])
p = torch.tensor([-.15,0.])
v = torch.stack((pt*p.cos(),pt*p.sin(),
    pt*y.sinh(),pt*y.cosh()), -1)[None]
u = pair_features(v, torch.ones(1,2).bool())
print(u[0,0,1].round(decimals=4))
""",
"layernorm": """import torch
x = torch.tensor([2.,4.,6.,8.])
g = torch.tensor([1.,2.,.5,1.])
z = (x-x.mean())/(x.var(unbiased=False)+1e-5).sqrt()
print((g*z+.1).round(decimals=3))
""",
}
