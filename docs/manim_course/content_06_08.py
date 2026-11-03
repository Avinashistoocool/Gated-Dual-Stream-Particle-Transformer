"""Original English narration and executable examples for Chapters 6–8.

All numerical examples are illustrative calculations, never measured results.
Sources: GDS-ParT report sections 3–4 and appendices C–I; executable code wins.
"""
SOURCE = "GDS-ParT_Technical_Report.tex: inputs/embeddings; data/jetclass_cache.py; gds_part_colab.py"


def block(caption, narration):
    return caption, narration, SOURCE


CHAPTER_06 = {
    "recap": block("Neural layers need observations before they can learn.",
        "Previously, we learned how tensors, linear maps, and nonlinear functions turn numbers into learned representations. But where do those numbers come from? Today we move upstream, from a reconstructed jet to its particle table. Every blue row must describe one particle, and every column must have a precise meaning."),
    "objective": block("You will construct all seventeen features and their mask.",
        "Our goal is to construct all seventeen constituent features, understand the ten jet labels, and explain clipping and padding. We will transform an explicit illustrative particle rather than merely list names. We will then execute the formulas and distinguish observed features from trainable parameters."),
    "jet": block("One jet collects particles but receives one target label.",
        "A high energy collision produces a collimated spray reconstructed as a jet. Each moving dot is an illustrative reconstructed constituent, not an actual event measurement. The enclosing jet receives one class label. Individual constituents are not independently labelled as Higgs bosons or top quarks by this classifier."),
    "classes": block("Ten labels describe simulated production and decay categories.",
        "The background category is QCD. Higgs categories are Hbb, Hcc, Hgg, H4q, and Hqql. The remaining categories are Zqq, Wqq, Tbqq, and Tbl. These distinguish processes and decay modes supplied by the dataset. A muon constituent alone cannot establish which of these ten processes generated its entire jet."),
    "decays": block("The category names abbreviate different decay products.",
        "QCD denotes quark or gluon background jets. Hbb, Hcc, and Hgg denote Higgs decays to bottom quarks, charm quarks, and gluons. H4q and Hqql use Higgs decays through W bosons, with four quarks or quarks plus a charged lepton and neutrino. Zqq and Wqq are hadronic boson decays; Tbqq and Tbl are hadronic and leptonic top categories."),
    "table": block("Seventeen columns describe each retained particle.",
        "Call the feature table F. Its shape is batch size B, particle slots N, and seventeen channels. Index i selects a particle and index c selects a feature. A separate four vector stores physical momentum and energy. We must not confuse this four vector with the first four transformed feature columns."),
    "logs": block("Logarithms compress positive momentum and energy scales.",
        "For feature zero, take the natural logarithm of transverse momentum, subtract one point seven, and multiply by zero point seven. Feature one uses energy and subtracts two instead. These constants are fixed preprocessing choices. Here numerical momenta and energies use the dataset's units, conventionally GeV; they are not learned weights."),
    "logs_numeric": block("Substitute momentum twenty and energy twenty five.",
        "For our illustrative particle, transverse momentum is twenty and energy is twenty five. The natural logarithm of twenty is approximately two point nine nine six. Subtracting one point seven and scaling gives zero point nine zero seven. Repeating the energy transformation gives approximately zero point eight five three."),
    "fractions": block("Relative features compare a particle with its whole jet.",
        "Features two and three use fractions. Divide particle transverse momentum by jet transverse momentum before taking the logarithm; then add four point seven and multiply by zero point seven. The energy fraction uses jet energy in its denominator. Using jet momentum one hundred and jet energy one hundred twenty five makes both fractions one fifth."),
    "fraction_numeric": block("The same fraction produces the same transformed value.",
        "The natural logarithm of one fifth is approximately minus one point six zero nine. Adding four point seven gives three point zero nine one. Multiplying by zero point seven gives two point one six three. Both relative features therefore agree in this example, although the two absolute logarithmic features do not."),
    "angles": block("Three columns locate the particle relative to the jet.",
        "Feature five is the supplied pseudorapidity offset from the jet axis, and feature six is its supplied azimuthal offset. Feature four first computes their Euclidean distance, subtracts zero point two, and multiplies by four. This constituent to jet distance uses pseudorapidity. Later pair distances use rapidity; these variables are not always identical."),
    "angles_numeric": block("Offsets three tenths and four tenths give distance one half.",
        "Choose pseudorapidity offset zero point three and azimuthal offset zero point four. Squaring gives zero point zero nine and zero point one six. Their sum is zero point two five, whose square root is zero point five. Thus feature four is four times zero point three, or one point two."),
    "identity": block("Charge and five flags describe constituent identity.",
        "Feature seven is electric charge in the dataset convention. Features eight through twelve are charged hadron, neutral hadron, photon, electron, and muon indicators, in exactly that order. Our illustrative positive charged hadron has charge one and flags one, zero, zero, zero, zero. These are constituent indicators, not the ten jet classes."),
    "tracks": block("Track displacements and uncertainties complete the feature vector.",
        "Features thirteen and fifteen apply hyperbolic tangent to transverse and longitudinal impact parameters. These describe displacement relative to a reference vertex. Features fourteen and sixteen clip their uncertainties between zero and one. The argument follows the dataset numerical convention. A displaced track can be informative, but does not prove a heavy flavour origin."),
    "tracks_numeric": block("Bounded transforms handle displacement and uncertainty differently.",
        "Choose transverse displacement zero point five, uncertainty zero point two, longitudinal displacement minus one, and uncertainty one point four. Hyperbolic tangent gives approximately zero point four six two and minus zero point seven six two. The first uncertainty stays zero point two. The last clips to one; applying hyperbolic tangent to uncertainties would be incorrect."),
    "clip": block("Clipping bounds every final feature between minus five and five.",
        "The cache stacks all seventeen columns and clips the complete array between minus five and five. Values seven and minus nine become five and minus five, while zero point four stays unchanged. Non finite cleanup follows clipping: NaN becomes zero. Infinities already clipped to the endpoints stay there. Features are stored in float sixteen."),
    "code": block("Executable PyTorch reproduces the complete illustrative feature row.",
        "The code assembles the complete illustrative feature row. Computed logarithms and hyperbolic tangents join the distance, identity flags, and clipped uncertainties we already calculated. It prints all seventeen entries in order, not random values with the right shape. Compare each group with our calculations. Four vectors remain separately stored in float thirty two."),
    "padding": block("Padding creates slots, never additional physical particles.",
        "The cache keeps at most the first one hundred twenty eight constituents, which are already transverse momentum ordered by the dataset. It stores them without padding. Each batch is padded only to its longest retained jet. For counts three and five, N is five, and the shorter jet needs two zero slots with false mask entries."),
    "mask": block("A mask distinguishes absent slots from genuine zero features.",
        "A valid particle can contain zero features, so testing whether a feature equals zero is not a validity rule. The explicit Boolean mask marks real constituents. For counts three and five, the first row is true, true, true, false, false. Invalid slots must not influence attention or pooling; a biased linear layer can make padded embeddings nonzero."),
    "exercise1": block("Exercise one: transform a particle with momentum ten.",
        "Compute feature zero when transverse momentum is ten. The natural logarithm is approximately two point three zero three. Subtract one point seven to obtain zero point six zero three. Multiply by zero point seven to obtain zero point four two two. No clipping is needed because this lies between minus five and five."),
    "exercise2": block("Exercise two: retain and pad jets of different lengths.",
        "A batch contains jets with one hundred thirty and three constituents. The first retains one hundred twenty eight, dropping two. Therefore the padded width is one hundred twenty eight. The shorter jet receives one hundred twenty five padding slots. Its mask has three true entries; do not silently classify padding as low energy particles."),
    "exercise3": block("Exercise three: distinguish a muon from a jet category.",
        "A constituent has its muon indicator equal to one. Does this determine the jet label Tbl? No. A constituent identity is not a process label, and several categories can contain leptons. The classifier combines the whole collection and pair relationships. Similarly, seventeen recorded columns do not mean seventeen trainable model parameters."),
    "summary": block("Observed features, retained particles, and masks have distinct roles.",
        "Keep three conclusions. Seventeen fixed feature recipes describe each retained constituent. Ten labels describe whole simulated jets, not individual particles. Padding supplies storage slots while masks preserve physical membership. On our component map, this chapter fills the input stage. Next we use separate four vectors to construct the orange pair inputs."),
}

CHAPTER_07 = {
    "recap": block("Particle observations and physical four vectors remain separate.",
        "We have built seventeen transformed features for each constituent, together with a validity mask. Four vectors were stored separately because physical pair calculations require momentum and energy, not arbitrary normalized columns. Now we connect two constituents with an orange edge and compute the four numbers that describe their relationship."),
    "objective": block("You will derive and calculate all four pair features.",
        "Our goal is to derive transverse momentum, azimuth, rapidity, invariant mass, and the four pair inputs. We will calculate one complete pair, execute the repository routine, and explain its numerical safeguards. We will also see why an arbitrary constituent pair does not reconstruct the full parent resonance."),
    "four": block("A four vector stores momentum components and energy.",
        "Using units where the speed of light is one, the code stores px, py, pz, and energy, in that order. The first two momentum components lie perpendicular to the beam. The third follows the beam. Energy is the last component here; changing the convention silently would corrupt every invariant mass calculation."),
    "pt": block("Transverse momentum is a length in the transverse plane.",
        "Project the momentum onto the plane perpendicular to the beam. Pythagoras gives transverse momentum as the square root of px squared plus py squared. With px equal to three and py equal to four, the two squares are nine and sixteen. The transverse magnitude is therefore five, regardless of the longitudinal component."),
    "phi": block("Azimuth requires both signs to select the correct quadrant.",
        "The azimuthal angle is atan2 of py and px. With components three and four, it is approximately zero point nine two seven radians. A plain inverse tangent of their ratio loses quadrant information. Opposite vectors can have the same ratio, while atan2 correctly distinguishes their directions around the beam axis."),
    "mass": block("Mass squared subtracts squared momentum from squared energy.",
        "For a single four vector, mass squared is energy squared minus all three squared momentum components. Choose components three, four, twelve, and energy thirteen. Momentum squared is nine plus sixteen plus one hundred forty four, or one hundred sixty nine. Energy squared is also one hundred sixty nine, so this illustrative particle is massless."),
    "rapidity": block("Rapidity uses energy and longitudinal momentum.",
        "Rapidity is one half the natural logarithm of energy plus pz divided by energy minus pz. For energy thirteen and pz twelve, the ratio is twenty five. Its logarithm is approximately three point two one nine, so rapidity is approximately one point six zero nine. The denominator must remain positive for this physical example."),
    "eta": block("Pseudorapidity describes direction rather than energy.",
        "Pseudorapidity is minus the logarithm of the tangent of half the polar angle. For a massless particle it equals rapidity, so our previous vector gives the same value. For a massive particle it need not. The constituent distance from Chapter Six used pseudorapidity; the pair routine explicitly computes rapidity from energy and pz."),
    "boost": block("Rapidity differences survive a common longitudinal boost.",
        "A longitudinal boost shifts every rapidity by the same amount. Subtracting two shifted rapidities cancels that common shift. Transverse momenta and invariant masses also remain unchanged. This motivates our pair variables, but does not prove invariance of the entire network: its other features include energies and track measurements."),
    "wrap": block("Angles near the branch cut need a wrapped difference.",
        "Consider angles three point one and minus three point one radians. Direct subtraction gives six point two, almost a full circle, although the directions are close. Adding pi, taking the remainder modulo two pi, and subtracting pi gives approximately minus zero point zero eight three. This chooses the short signed difference."),
    "delta": block("Pair distance combines rapidity and wrapped azimuth differences.",
        "Call the wrapped angle difference delta phi and the rapidity difference delta y. The pair distance Delta is the square root of their squared sum. For delta y equal to zero point one five and delta phi equal to minus zero point one five, the squared sum is zero point zero four five. Delta is zero point two one two one three."),
    "kt": block("Relative transverse scale uses the softer particle.",
        "The pair's k T is the smaller transverse momentum multiplied by Delta. Our two massless constituents have transverse momenta eighty and thirty. The smaller value is thirty, so multiply thirty by zero point two one two one three. This gives approximately six point three six four GeV. Using the larger momentum would define a different feature."),
    "z": block("Momentum sharing divides the smaller momentum by their sum.",
        "The sharing fraction z divides the smaller transverse momentum by the sum of both. Thirty divided by eighty plus thirty is thirty over one hundred ten, approximately zero point two seven two seven three. For positive momenta, this definition never exceeds one half. It is symmetric even though we call one particle softer."),
    "pairmass": block("Add four vectors before computing the pair invariant mass.",
        "The pair mass comes from the sum of both four vectors, not the sum of their individual masses. Expanding the invariant for two massless constituents yields twice their transverse momentum product times cosh of delta y minus cosine of delta phi. Cross terms are essential; both individual masses can vanish while the pair mass is positive."),
    "pairmass_numeric": block("The complete example gives mass squared about one hundred eight.",
        "Substitute eighty, thirty, and the two angle differences of magnitude zero point one five. Cosh is approximately one point zero one one two seven one, and cosine is approximately zero point nine eight eight seven seven one. Their difference times four thousand eight hundred gives one hundred eight point zero zero zero two GeV squared."),
    "logs": block("Four natural logarithms become the pair input vector.",
        "Take natural logarithms only after computing the four physical quantities. In order, the vector is log Delta, log k T, log z, and log mass squared. Our numerical result is approximately minus one point five five zero five, one point eight five zero seven, minus one point two nine nine three, and four point six eight two one."),
    "floors": block("Floors prevent logarithms of zero for physical finite inputs.",
        "The routine floors relevant intermediate quantities at ten to the minus eight. It then sets self pairs and padding pairs to zero. A zero feature vector here means an explicitly invalid relation, not the logarithm of four unit quantities. Pair features are symmetric under exchanging the particles; the full matrix stores both directions."),
    "code": block("Executed code checks the full pair and its validity mask.",
        "We construct the two massless four vectors from transverse momentum, rapidity, and azimuth, then call the actual pairwise features routine. It internally converts to float thirty two, so the final decimals can differ slightly from the double precision hand derivation. The output also verifies zero diagonal entries and a zero padded row."),
    "difference": block("Report vs code difference: float precision changes final decimals.",
        "The report's illustrative mass calculation uses rounded analytic values. The code explicitly casts four vectors to float thirty two before subtraction, so cancellation affects the final few decimals. We preserve the report's analytic example and show the actual code output separately. This is a numerical precision difference, not a different physics definition."),
    "resonance": block("An arbitrary constituent pair is not the whole parent.",
        "Two final state constituents usually capture only a small portion of the decay system. Their invariant mass therefore does not generally peak at the Higgs or W mass. A resonance interpretation requires the appropriate full decay system or suitable reconstructed subjets. Pair features provide structure without identifying a unique physical splitting."),
    "exercise1": block("Exercise one: exchange the two constituent addresses.",
        "Exchange particles A and B. Rapidity and wrapped azimuth differences change sign, but their squares do not. The minimum, sum, and combined four vector are unchanged. Consequently all four pair features stay the same. The orange table is symmetric even though the row and column addresses are different."),
    "exercise2": block("Exercise two: equal transverse momenta fix the sharing fraction.",
        "Give both particles transverse momentum forty and separation zero point two. The smaller momentum is forty, their sum is eighty, and z equals one half. The relative transverse scale is forty times zero point two, or eight. Their logarithms are approximately minus zero point six nine three and two point zero seven nine, respectively."),
    "exercise3": block("Exercise three: count valid ordered pairs with padding.",
        "A jet has three real constituents but five padded slots. The pair tensor contains twenty five address pairs. Only three times two, or six, are real nonself ordered pairs. Three diagonal entries and every padding relation are zeroed. Counting unordered physical connections instead gives three, which is not the stored tensor entry count."),
    "summary": block("Four vectors produce symmetric, masked, logarithmic pair features.",
        "Keep three conclusions. Four vectors determine transverse momentum, rapidity, azimuth, and invariant mass. Four symmetric logarithmic features describe each valid pair. Floors and masks make the finite input calculation usable without turning constituent masses into parent resonance claims. Our component map now contains both blue particle inputs and orange pair inputs."),
}

CHAPTER_08 = {
    "recap": block("Blue particle inputs and orange pair inputs are ready.",
        "We now have seventeen features per constituent and four features per constituent pair. Their axes already tell us that these are different objects. This chapter turns both inputs into learned representations, derives LayerNorm completely, and explains residual connections and dropout without pretending that extra channels are extra detector measurements."),
    "objective": block("You will calculate normalization, embeddings, and parameter counts.",
        "Our objective is to compute LayerNorm by hand and in PyTorch, follow both embedding modules, and count their trainable parameters. We will also calculate a residual update and an illustrative dropout mask. Finally we will trace the two representations through several blocks to pin down exactly what dual stream means."),
    "embedding": block("An embedding learns a useful coordinate system.",
        "The particle embedding maps seventeen input channels to one hundred twenty eight learned channels. It applies the same weights to every particle. Increasing the width does not create one hundred eleven new measurements. Each output coordinate can mix multiple inputs, and there is no guarantee that a particular coordinate retains a simple physical name."),
    "ln_problem": block("LayerNorm compares channels within one particle.",
        "Inputs can have different offsets and scales. LayerNorm computes statistics across the final channel axis independently for each particle. It does not average different jets together and does not normalize each channel across the batch. To see the operation clearly, use the four channel illustrative vector two, four, six, eight."),
    "ln_mean": block("Subtract the channel mean before measuring spread.",
        "Add two, four, six, and eight to obtain twenty. Divide by four channels to obtain mean five. Subtract five from each entry to obtain minus three, minus one, one, and three. These deviations sum to zero. Their signs record whether each original channel lies below or above this particle's mean."),
    "ln_variance": block("LayerNorm variance divides squared deviations by four.",
        "Square the deviations to obtain nine, one, one, and nine. Their sum is twenty. Divide by the channel count four, not three, giving variance five. PyTorch LayerNorm uses this population variance convention. A small epsilon is added before taking the square root to protect constant vectors from division by zero."),
    "ln_normal": block("Normalize each deviation by the same positive scale.",
        "The square root of five is approximately two point two three six. Dividing each deviation by this value gives approximately minus one point three four two, minus zero point four four seven, zero point four four seven, and one point three four two. These displayed values ignore the tiny effect of epsilon in the final decimals."),
    "ln_affine": block("Learned scale and shift act independently on each channel.",
        "Now multiply elementwise by gamma equal to one, two, one half, one, and add beta equal to zero point one in every channel. The result is approximately minus one point two four two, minus zero point seven nine four, zero point three two four, and one point four four two. Neither zero mean nor unit variance is guaranteed after this affine transformation."),
    "ln_code": block("Executed LayerNorm confirms every step with epsilon included.",
        "This code computes the mean, population variance, and affine LayerNorm using epsilon ten to the minus five. The captured output comes from actual PyTorch execution. Notice that gamma multiplies corresponding channels rather than taking an inner product. The output retains four channels, and LayerNorm has eight trainable scalars in this small example."),
    "particle": block("The particle module normalizes before its two affine maps.",
        "The actual module first applies LayerNorm across seventeen channels. A linear layer maps seventeen to one hundred twenty eight, GELU supplies a nonlinear response, and another linear layer keeps width one hundred twenty eight. Dropout follows that second map. There is no residual addition around this entire input embedding."),
    "particle_toy": block("A tiny embedding makes every learned arithmetic step visible.",
        "Use an explicitly illustrative two channel embedding, not trained model weights. Normalize two and four to approximately minus one and one. A diagonal first weight matrix with entries one and two gives minus one and two. GELU produces minus zero point one five eight seven and one point nine five four five. The displayed second matrix and bias give one point eight nine five eight and minus two point four seven one eight."),
    "toy_code": block("Executed code verifies the complete tiny embedding.",
        "This code follows the tiny embedding from its original input to its final output, including LayerNorm epsilon and GELU. The diagonal multiplication is just a compact way to apply the displayed first matrix. The at sign applies the second matrix. These outputs are consequences of deliberately chosen weights, not experimental model measurements."),
    "parameter": block("Count weights, biases, and normalization parameters separately.",
        "LayerNorm contributes seventeen scales and seventeen shifts, giving thirty four. The first affine map contributes seventeen times one hundred twenty eight weights plus one hundred twenty eight biases, giving two thousand three hundred four. The second contributes sixteen thousand five hundred twelve. Adding all three terms gives eighteen thousand eight hundred fifty trainable parameters."),
    "pair": block("The pair module preserves both constituent addresses.",
        "The pair embedding applies a four to sixty four linear map, GELU, and a sixty four to sixty four linear map independently to every pair. It contains neither LayerNorm nor dropout. Shared weights do not depend on either particle address. The result retains both N axes, so its shape is B, N, N, sixty four."),
    "pair_toy": block("A tiny pair embedding keeps the relation interpretation.",
        "For a fully calculable pair example, use the four log features from Chapter Seven. Choose a first weight row one, zero, zero, zero with zero bias, so the hidden scalar is minus one point five five zero five. GELU gives approximately minus zero point zero nine three eight. A second weight two and bias zero point one produce approximately minus zero point zero eight seven six."),
    "pair_count": block("Count the two pair embedding layers separately.",
        "The first pair map contains four times sixty four weights and sixty four biases, totaling three hundred twenty. The second contains sixty four squared weights and sixty four biases, totaling four thousand one hundred sixty. Their sum is four thousand four hundred eighty. This count does not grow with the number of particle pairs."),
    "embedding_code": block("Executing the actual modules verifies both shapes and counts.",
        "We instantiate the repository's particle and pair modules, switch to evaluation mode, and provide two jets with three particle slots each. Actual execution returns particle shape two, three, one hundred twenty eight and relation shape two, three, three, sixty four. Summing parameter sizes confirms both hand counts without inventing training measurements."),
    "residual": block("A residual connection adds a same shaped learned update.",
        "After embedding, transformer blocks preserve a particle state while adding an update. For a simple illustrative state two, minus one and update zero point five, three, elementwise addition gives two point five, two. Both paths must have the same shape. A residual connection is addition, not concatenation, and does not introduce its own trainable matrix."),
    "pre_norm": block("Pre normalization leaves the residual path outside LayerNorm.",
        "The particle block normalizes the current state before attention, then adds the attention update to the original state. It repeats that pattern for the feed forward update with a second normalization. This is pre normalization. Replacing the result by only its update would remove the shortcut and implement a different architecture."),
    "dropout": block("Training dropout rescales surviving activations.",
        "For dropout probability one half, an illustrative keep mask one, zero, one, zero applied to two, four, six, eight gives four, zero, twelve, zero. Surviving entries are divided by one minus the dropout probability. This is one possible mask, not a measured model output. In evaluation mode dropout returns its input unchanged."),
    "dropout_code": block("A dropout realization differs from its expectation.",
        "The code explicitly supplies our illustrative keep mask rather than claiming a particular random draw. Averaging over all sixteen equally likely masks recovers the original vector. A single realization need not have the same sum or mean. Evaluation mode disables random dropping; it is not another training sample."),
    "dual": block("Two streams mean particle states and pair relations.",
        "The blue representation X has shape B, N, D. The orange representation R has shape B, N, N, D sub p. These are the two streams. Queries and keys are projections used inside particle attention, not the two streams themselves. Orange relations influence red gates and teal pair bias, while values remain yellow."),
    "reuse": block("Compute the relation representation once and reuse it across blocks.",
        "The forward pass computes R once before the particle block loop. Blue particle states evolve from one block to the next, while the same orange tensor feeds multiple blocks. State conditioned relation summaries can change as X changes, but that does not mean R is recurrently rewritten. Reuse applies within one forward pass, not across unrelated batches."),
    "exercise1": block("Exercise one: normalize a constant vector.",
        "Take the vector one, one, one with gamma one and beta zero. Its mean is one, all deviations are zero, and its variance is zero. Epsilon makes the denominator positive, so each normalized output is zero. Omitting epsilon would create an undefined zero divided by zero instead of this well defined result."),
    "exercise2": block("Exercise two: count a seventeen to one hundred twenty eight map.",
        "Count only one affine layer, not the entire embedding. There are seventeen times one hundred twenty eight, or two thousand one hundred seventy six, weights. Adding one hundred twenty eight biases gives two thousand three hundred four parameters. Do not multiply this total by the number of particles: the same layer is shared across every slot."),
    "exercise3": block("Exercise three: distinguish a reused tensor from changing summaries.",
        "Suppose the first and second blocks use different particle states to summarize the same relation tensor. Can their summaries differ? Yes, because the weighting rule may depend on the current state. Does that prove R changed? No. Changing how a fixed table is read is not the same as rewriting the table."),
    "summary": block("Normalization, embedding, and residual updates have different jobs.",
        "Keep three conclusions. LayerNorm standardizes each particle across channels before a learned affine adjustment. Separate embeddings produce blue particle states and orange pair relations with different axes. Residual updates evolve X while R is reused within the forward pass. We are now ready to derive attention without confusing preprocessing, learned coordinates, and physical measurements."),
}

NARRATION = {6: CHAPTER_06, 7: CHAPTER_07, 8: CHAPTER_08}

CODE = {
    "c06_features": """from torch import tensor as t
a = (.7*(t([20.,25.]).log()-t([1.7,2.]))).tolist()
r = float(.7*(t(.2).log()+4.7))
d,z = t([.5,-1.]).tanh().tolist()
f = a+[r,r,1.2,.3,.4,1,1,0,0,0,0,d,.2,z,1]
for row in (f[:6],f[6:12],f[12:]):
    print([round(v,3) for v in row])""",
    "c07_pair": """import torch
import data.jetclass_cache as jc
p,y,f = map(torch.tensor,([80,30],[.15,0],[-.15,0]))
v = torch.stack([p*f.cos(),p*f.sin(),
                 p*y.sinh(),p*y.cosh()],-1)
v = torch.cat([v,torch.zeros(1,4)])[None]
u = jc.pairwise_features(v,torch.tensor([[1,1,0]])>0)
print(u[0,0,1].round(decimals=4))
print(u[0,0,0], u[0,2].sum())""",
    "c08_layernorm": """import torch
x = torch.tensor([2.,4.,6.,8.])
g = torch.tensor([1.,2.,.5,1.])
b = torch.full((4,),.1)
print(x.mean(), x.var(unbiased=False))
y = torch.nn.functional.layer_norm(
    x,(4,),g,b,eps=1e-5)
print(y.round(decimals=3))""",
    "c08_embeddings": """import torch, gds_part_colab as g
a = g.ParticleEmbedding(17,128).eval()
b = g.PairwiseEmbedding(4,64).eval()
print(tuple(a(torch.zeros(2,3,17)).shape))
print(tuple(b(torch.zeros(2,3,3,4)).shape))
count = lambda m: sum(p.numel() for p in m.parameters())
print(count(a), count(b))""",
    "c08_toy": """import torch
F = torch.nn.functional
x = torch.tensor([2.,4.])
n = F.layer_norm(x,(2,))
h = F.gelu(n*torch.tensor([1.,2.]))
W = torch.tensor([[1.,1.],[2.,-1.]])
print(W@h+torch.tensor([.1,-.2]))
print(2*F.gelu(torch.tensor(-1.5505))+.1)""",
    "c08_dropout": """import torch, itertools as it
x = torch.tensor([2.,4.,6.,8.])
mask = torch.tensor([1.,0.,1.,0.])
print(x*mask/.5)
masks = torch.tensor([*it.product([0.,1.],repeat=4)])
print((x*masks/.5).mean(0))
print(torch.nn.Dropout(.5).eval()(x))""",
}

# Equations follow the spoken derivation; earlier steps remain as visual history.
STEPS = {
6: {
"recap": (r"F\longrightarrow X\longrightarrow\text{jet label}", r"F_{b,i,c}:\text{observations, not weights}"),
"objective": (r"F:(B,N,17)",r"m:(B,N),\quad N\le128",r"y_{\rm class}\in\{0,\ldots,9\}"),
"jet": (r"\{p_1,\ldots,p_n\}\longrightarrow\text{one jet}",r"\text{one jet}\longrightarrow\text{one label}"),
"classes": (r"\mathrm{QCD}",r"\mathrm{Hbb,\ Hcc,\ Hgg,\ H4q,\ Hqql}",r"\mathrm{Zqq,\ Wqq,\ Tbqq,\ Tbl}"),
"decays": (r"\mathrm{QCD}:q/g;\quad H\to b\bar b,\ c\bar c,\ gg",r"H\to WW^*\to4q\ \text{or}\ qq\ell\nu",r"Z/W\to qq;\quad t\to bqq\ \text{or}\ b\ell\nu"),
"table": (r"F_{b,i,:}\in\mathbb R^{17}",r"v_{b,i,:}=(p_x,p_y,p_z,E)\in\mathbb R^4",r"F:(B,N,17),\quad v:(B,N,4)"),
"logs": (r"f_0=0.7(\ln p_T-1.7)",r"f_1=0.7(\ln E-2.0)",r"\text{shift and scale: fixed constants}"),
"logs_numeric": (r"p_T=20,\quad E=25",r"f_0=0.7(2.995732-1.7)",r"f_0=0.907012,\quad f_1=0.853213"),
"fractions": (r"f_2=0.7[\ln(p_T/p_{T,\rm jet})+4.7]",r"f_3=0.7[\ln(E/E_{\rm jet})+4.7]",r"20/100=25/125=0.2"),
"fraction_numeric": (r"\ln(0.2)=-1.609438",r"0.7(-1.609438+4.7)=2.163393",r"(f_2,f_3)=(2.163393,2.163393)"),
"angles": (r"f_5=\Delta\eta,\quad f_6=\Delta\phi",r"r=\sqrt{\Delta\eta^2+\Delta\phi^2}",r"f_4=4(r-0.2)"),
"angles_numeric": (r"(\Delta\eta,\Delta\phi)=(0.3,0.4)",r"r=\sqrt{0.09+0.16}=0.5",r"(f_4,f_5,f_6)=(1.2,0.3,0.4)"),
"identity": (r"f_7=q",r"(f_8,\ldots,f_{12})=(I_{\rm ch.had},I_{\rm neut.had},I_\gamma,I_e,I_\mu)",r"(f_7,\ldots,f_{12})=(1,1,0,0,0,0)"),
"tracks": (r"f_{13}=\tanh d_0,\quad f_{15}=\tanh d_z",r"f_{14}=\operatorname{clip}(\sigma_{d_0},0,1)",r"f_{16}=\operatorname{clip}(\sigma_{d_z},0,1)"),
"tracks_numeric": (r"(d_0,\sigma_{d_0},d_z,\sigma_{d_z})=(0.5,0.2,-1,1.4)",r"(\tanh0.5,\tanh(-1))=(0.462117,-0.761594)",r"(f_{13},f_{14},f_{15},f_{16})=(0.462117,0.2,-0.761594,1)"),
"clip": (r"\operatorname{clip}(x,-5,5)=\min(5,\max(-5,x))",r"(7,-9,0.4)\longrightarrow(5,-5,0.4)",r"\mathrm{NaN}\to0,\quad+\infty\to5,\quad-\infty\to-5"),
"code": ("c06_features",),
"padding": (r"(n_1,n_2)=(3,5)\longrightarrow N=5",r"n_i^{\rm retained}=\min(n_i,128)",r"F:(2,5,17)"),
"mask": (r"m=\begin{bmatrix}1&1&1&0&0\\1&1&1&1&1\end{bmatrix}",r"0\text{ feature}\ne0\text{ validity}",r"W0+b=b\ne0\text{ generally}"),
"exercise1": (r"p_T=10:\quad f_0=0.7(\ln10-1.7)",r"=0.7(2.302585-1.7)",r"=0.421810"),
"exercise2": (r"(130,3)\longrightarrow(128,3)",r"N=128,\quad\text{dropped}=2",r"\text{shorter jet padding}=128-3=125"),
"exercise3": (r"I_\mu=1\ \not\Rightarrow\ y_{\rm class}=\mathrm{Tbl}",r"17\text{ inputs}\ne17\text{ parameters}"),
"summary": (r"17\text{ features per constituent}",r"10\text{ labels per jet}",r"m:\text{real slots only}"),
},
7: {
"recap": (r"F:(B,N,17)",r"v:(B,N,4)",r"U:(B,N,N,4)"),
"objective": (r"v_i,v_j\longrightarrow(\Delta,k_T,z,m^2)",r"U_{ij}=(\ln\Delta,\ln k_T,\ln z,\ln m^2)"),
"four": (r"c=1:\quad v=(p_x,p_y,p_z,E)",r"v=(3,4,12,13)",r"\text{energy is the final stored coordinate}"),
"pt": (r"p_T^2=p_x^2+p_y^2",r"p_T=\sqrt{3^2+4^2}",r"p_T=\sqrt{25}=5"),
"phi": (r"\phi=\operatorname{atan2}(p_y,p_x)",r"\phi=\operatorname{atan2}(4,3)=0.927295",r"\operatorname{atan2}(-4,-3)=-2.214297"),
"mass": (r"m^2=E^2-p_x^2-p_y^2-p_z^2",r"m^2=13^2-3^2-4^2-12^2",r"m^2=169-9-16-144=0"),
"rapidity": (r"y=\frac12\ln\frac{E+p_z}{E-p_z}",r"y=\frac12\ln\frac{13+12}{13-12}",r"y=\frac12\ln25=1.609438"),
"eta": (r"\eta=-\ln\tan(\theta/2)",r"m=0\Longrightarrow y=\eta",r"m>0:\quad y\ne\eta\text{ generally}"),
"boost": (r"y_i'=y_i-a,\quad y_j'=y_j-a",r"y_i'-y_j'=(y_i-a)-(y_j-a)",r"\Delta y'=\Delta y"),
"wrap": (r"\phi_i=3.1,\quad\phi_j=-3.1",r"\phi_i-\phi_j=6.2",r"\delta\phi=((6.2+\pi)\bmod2\pi)-\pi=-0.083185"),
"delta": (r"\Delta=\sqrt{(y_i-y_j)^2+\delta\phi^2}",r"\Delta=\sqrt{0.15^2+(-0.15)^2}",r"\Delta=\sqrt{0.045}=0.212132"),
"kt": (r"(p_{T,i},p_{T,j})=(80,30)",r"k_T=\min(80,30)\Delta",r"k_T=30(0.212132)=6.363961"),
"z": (r"z=\frac{\min(p_{T,i},p_{T,j})}{p_{T,i}+p_{T,j}}",r"z=\frac{30}{80+30}=0.272727",r"0<z\le\frac12"),
"pairmass": (r"m_{ij}^2=(E_i+E_j)^2-\|\mathbf p_i+\mathbf p_j\|^2",r"m_i=m_j=0:\quad m_{ij}^2=2(E_iE_j-\mathbf p_i\cdot\mathbf p_j)",r"m_{ij}^2=2p_{T,i}p_{T,j}[\cosh\Delta y-\cos\delta\phi]"),
"pairmass_numeric": (r"m^2=4800[\cosh0.15-\cos(-0.15)]",r"=4800(1.0112711096-0.9887710779)",r"m^2=108.0001519\approx108.0002\ {\rm GeV}^2"),
"logs": (r"U=(\ln0.212132,\ln6.363961,\ln0.272727,\ln108.000152)",r"U\approx(-1.5505,\ 1.8507,\ -1.2993,\ 4.6821)"),
"floors": (r"\epsilon=10^{-8},\quad\ln\max(q,\epsilon)",r"v_{ij}=m_im_j\mathbf1[i\ne j]",r"U_{ij}\leftarrow v_{ij}U_{ij}"),
"code": ("c07_pair",),
"difference": (r"v\leftarrow v.\mathrm{float}()\quad\text{in executable code}",r"m^2_{\rm analytic}=108.0001519",r"U_{\rm float32}\approx U_{\rm analytic}\quad\text{within rounding}"),
"resonance": (r"m_{\rm two\ arbitrary\ constituents}\ne m_{\rm full\ parent}",r"m_{\rm appropriate\ full\ decay\ system}:\text{resonance information}"),
"exercise1": (r"(i,j)\leftrightarrow(j,i):\quad(\Delta y,\delta\phi)\to(-\Delta y,-\delta\phi)",r"\Delta,\min(p_{T,i},p_{T,j}),p_i+p_j\text{ unchanged}",r"U_{ij}=U_{ji}"),
"exercise2": (r"p_{T,i}=p_{T,j}=40,\quad\Delta=0.2",r"z=40/80=0.5,\quad k_T=40(0.2)=8",r"\ln z=-0.693147,\quad\ln k_T=2.079442"),
"exercise3": (r"N=5,\quad n_{\rm real}=3",r"N^2=25,\quad n_{\rm ordered}=3(3-1)=6",r"n_{\rm unordered}=6/2=3"),
"summary": (r"v_i,v_j\longrightarrow\text{physical pair quantities}",r"U=(\ln\Delta,\ln k_T,\ln z,\ln m^2)",r"U_{ij}=U_{ji},\quad U_{ii}=0"),
},
8: {
"recap": (r"F:(B,N,17),\quad U:(B,N,N,4)",r"F\longrightarrow X,\quad U\longrightarrow R"),
"objective": (r"\mathrm{LayerNorm}\longrightarrow\mathrm{embeddings}",r"\mathrm{residuals,\ dropout,\ parameter\ counts}"),
"embedding": (r"f_i\in\mathbb R^{17}\longrightarrow x_i\in\mathbb R^{128}",r"x_i=Wf_i+b:\text{learned coordinates}",r"128-17=111\text{ channels, not new measurements}"),
"ln_problem": (r"X:(B,N,D),\quad\mathrm{LN}\text{ reduces only }D",r"x=(2,4,6,8),\quad D=4"),
"ln_mean": (r"\mu=\frac1D\sum_cx_c",r"\mu=(2+4+6+8)/4=5",r"x-\mu=(-3,-1,1,3)"),
"ln_variance": (r"(x-\mu)^2=(9,1,1,9)",r"\sigma^2=(9+1+1+9)/4=5",r"s=\sqrt{\sigma^2+\epsilon}=\sqrt{5+\epsilon}"),
"ln_normal": (r"\widehat x_c=(x_c-\mu)/\sqrt{\sigma^2+\epsilon}",r"\widehat x=(-3,-1,1,3)/\sqrt{5+\epsilon}",r"\widehat x\approx(-1.342,-0.447,0.447,1.342)"),
"ln_affine": (r"\gamma=(1,2,0.5,1),\quad\beta=(0.1,0.1,0.1,0.1)",r"\mathrm{LN}(x)=\gamma\odot\widehat x+\beta",r"\mathrm{LN}(x)\approx(-1.242,-0.794,0.324,1.442)"),
"ln_code": ("c08_layernorm",),
"particle": (r"f_i\longrightarrow\mathrm{LN}_{17}(f_i)",r"h_i=\mathrm{GELU}(W_1\mathrm{LN}_{17}(f_i)+b_1)",r"x_i=\mathrm{Dropout}(W_2h_i+b_2)"),
"particle_toy": (r"f=(2,4)\longrightarrow\mathrm{LN}(f)\approx(-1,1)",r"W_1=\begin{bmatrix}1&0\\0&2\end{bmatrix},\quad b_1=0",r"h=\mathrm{GELU}(-1,2)\approx(-0.1587,1.9545)",r"x=\begin{bmatrix}1&1\\2&-1\end{bmatrix}h+\begin{bmatrix}0.1\\-0.2\end{bmatrix}\approx\begin{bmatrix}1.8958\\-2.4718\end{bmatrix}"),
"toy_code": ("c08_toy",),
"parameter": (r"\mathrm{LN}:2(17)=34",r"W_1,b_1:17(128)+128=2304",r"W_2,b_2:128^2+128=16512",r"34+2304+16512=\boxed{18850}"),
"pair": (r"U_{ij}\in\mathbb R^4\longrightarrow h_{ij}=\mathrm{GELU}(W_3U_{ij}+b_3)",r"r_{ij}=W_4h_{ij}+b_4\in\mathbb R^{64}",r"R:(B,N,N,64)\quad\text{no LN or dropout}"),
"pair_toy": (r"U=(-1.5505,1.8507,-1.2993,4.6821)",r"W_3=(1,0,0,0),\ b_3=0\Longrightarrow W_3U=-1.5505",r"h=\mathrm{GELU}(-1.5505)\approx-0.0938",r"r=2h+0.1\approx-0.0876"),
"pair_count": (r"4(64)+64=320",r"64^2+64=4160",r"320+4160=\boxed{4480}"),
"embedding_code": ("c08_embeddings",),
"residual": (r"x_{\rm new}=x+u,\quad x,u\in\mathbb R^D",r"x=(2,-1),\quad u=(0.5,3)",r"x_{\rm new}=(2+0.5,-1+3)=(2.5,2)"),
"pre_norm": (r"x'=x+\operatorname{Attn}(\mathrm{LN}_1(x),R)",r"x''=x'+\operatorname{FFN}(\mathrm{LN}_2(x'))",r"\text{shortcut outside normalization}"),
"dropout": (r"p=0.5,\quad k=(1,0,1,0)",r"\mathrm{Drop}(x)=k\odot x/(1-p)",r"(2,4,6,8)\longrightarrow(4,0,12,0)",r"\mathbb E[\mathrm{Drop}(x)]=x"),
"dropout_code": ("c08_dropout",),
"dual": (r"X:(B,N,D)\quad\text{particle stream}",r"R:(B,N,N,D_p)\quad\text{relation stream}",r"(X,R)\ne(Q,K)"),
"reuse": (r"R=\mathrm{PairEmbed}(U)\quad\text{once per forward pass}",r"X^{(\ell+1)}=\mathrm{Block}_{\ell}(X^{(\ell)},R)",r"R\text{ reused};\quad\mathrm{summary}(R,X^{(\ell)})\text{ may change}"),
"exercise1": (r"x=(1,1,1)\Longrightarrow\mu=1,\ \sigma^2=0",r"\widehat x=(0,0,0)/\sqrt{\epsilon}",r"\gamma=1,\beta=0\Longrightarrow\mathrm{LN}(x)=(0,0,0)"),
"exercise2": (r"17\to128:\quad\#W=17(128)=2176",r"\#b=128",r"\#\mathrm{parameters}=2176+128=2304"),
"exercise3": (r"s^{(1)}=s(R,X^{(1)}),\quad s^{(2)}=s(R,X^{(2)})",r"X^{(1)}\ne X^{(2)}\Longrightarrow s^{(1)}\text{ may differ from }s^{(2)}",r"R\text{ unchanged}"),
"summary": (r"\mathrm{LN}:\text{statistics across channels}",r"F\to X\text{ blue},\quad U\to R\text{ orange}",r"X\text{ evolves};\quad R\text{ reused}"),
}}

from course_content import b

CHAPTERS = {}
for number, narration in NARRATION.items():
    CHAPTERS[number] = {}
    for key, (caption, text, source) in narration.items():
        kind = "code" if key in {"code", "ln_code", "embedding_code", "dropout_code", "toy_code"} else "derive"
        color = "R" if number == 7 or key in {"pair", "pair_count", "reuse"} else "X"
        CHAPTERS[number][key] = b(caption, text, *STEPS[number][key],
                                 kind=kind, color=color, source=source)

EXTRA_SCREEN_TEXT = []
EXTRA_FORMULAS = [
    r"F:(B,N,17)\quad m:(B,N)",
    r"\{p_i\}_{i=1}^{n}\longrightarrow y_{\rm jet}",
    r"\Delta\phi=0.4", r"\Delta\eta=0.3",
    r"\sqrt{0.3^2+0.4^2}=0.5", r"f_4=4(0.5-0.2)=1.2",
    r"v:(B,N,4)\quad U:(B,N,N,4)",
    r"p_x=3", r"p_y=4", r"p_T^2=3^2+4^2", r"p_T=\sqrt{25}=5",
    r"\Delta\phi_{\rm naive}=6.2", r"\delta\phi=-0.083185",
    r"X:(B,N,128)\quad R:(B,N,N,64)",
    r"X^{(0)}", r"X^{(1)}", r"X^{(2)}",
    r"R:(B,N,N,64)", r"X:(B,N,128)", r"(X,R)\ne(Q,K)",
    r"R\text{ is computed once per forward pass}",
]
