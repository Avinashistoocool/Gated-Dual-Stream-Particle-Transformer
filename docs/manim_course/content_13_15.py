"""Original narration for chapters 13--15; sources name report sections.

The matching scene files own all diagrams. Numerical demonstrations are
illustrative calculations, never invented training measurements.
"""

CHAPTER_13 = {
    "recap": (
        "Two streams now become one complete prediction.",
        "We have built particle embeddings, pair embeddings, query and key gates, attention, and residual updates. Blue particle states change as information moves between particles. Orange relation states describe pairs. Their embedding is computed once. Now we must connect these components without silently changing the model.",
        "Report sections 4, 7, 11; gds_part_colab.py"),
    "objective": (
        "Trace every state from input to ten output scores.",
        "Our objective is a complete forward pass, including the pooling operation that converts many particles into one jet. We will track shapes, calculate residual and pooling examples, and test reordering in executable code. At each junction, ask which tensor changes and which tensor is reused.",
        "Report section 11"),
    "inputs": (
        "Name every axis before connecting the modules.",
        "B counts jets in the batch. N counts padded particle slots. D is the particle width, one hundred twenty eight. The relation width is sixty four, and there are eight attention heads. F contains seventeen features per particle. U contains four features per ordered pair. The Boolean mask marks real particles.",
        "Report sections 2, 3, 11"),
    "embed": (
        "Embed particles and pairs using separate learned functions.",
        "The particle embedding maps the final axis from seventeen to one hundred twenty eight. The pair embedding maps four to sixty four. Neither operation mixes jets or changes the number of particle slots. The two streams are different tensors with different jobs. A pair tensor has two particle axes, not one.",
        "Report sections 4, 11"),
    "reuse": (
        "Relations remain fixed while particle representations evolve.",
        "Create R once, before the particle blocks. Project the same R to eight additive bias channels and rearrange those channels into the head axis. In shared mode, also create the two gates once. Each block has its own particle projections and feed forward weights, but reuses R and the pair bias.",
        "Report section 11; GDSParT._pair_inputs"),
    "normalise": (
        "Normalize the current state before computing its update.",
        "At layer ell, X means the state entering this block, not the original measurements. Layer normalization produces H with the same shape. Attention reads H, but its result is added to X. Normalizing the residual path itself would be a different recurrence. The superscript labels a layer, not a power.",
        "Report section 11.1; GDSParTBlock.forward"),
    "gates": (
        "Configuration determines whether gates change between blocks.",
        "In shared mode, every block receives the same query and key gates. In conditioned mode, this block uses its normalized particle state H to summarize the fixed relations differently. Each block then has its own gate module. Fixed R therefore does not imply fixed gates in the conditioned configuration.",
        "Report sections 7, 11.1"),
    "attention": (
        "Gates reshape comparisons; bias adjusts individual pair scores.",
        "Green queries and purple keys are multiplied channel by channel by their red gates. Their dot product is divided by the square root of the head width, which is four here. Add the teal pair bias, exclude padded keys, and apply softmax over keys. Yellow values are not gated in this model.",
        "Report sections 5--9, 11.1"),
    "residual": (
        "Two residual additions complete one particle block.",
        "Attention produces a message with the same shape as X. Add it to the old state to obtain X tilde. Apply a second layer normalization to X tilde, then a feed forward network, and add that result to X tilde. The second branch must read the updated state, not the original state.",
        "Report section 11.1"),
    "residual_example": (
        "Calculate the updates before repeating the block.",
        "Here is an illustrative two channel calculation, not fitted model output. Begin with two and minus one. Suppose attention returns one half and one. Their sum is two point five and zero. Suppose the feed forward branch then returns minus one half and two. The next state is two and two.",
        "Derived worked example of report section 11.1"),
    "repeat": (
        "Repeat eight blocks without recomputing the relation embedding.",
        "Carry the new particle state into the next block. Repeat eight times for the reported shared version two model. The relation tensor follows the orange reuse path. It is not updated alongside the blue particle stream. After the eighth block, stop changing particle states and begin pooling.",
        "Report sections 3, 11"),
    "cls": (
        "A learned class token asks about the whole jet.",
        "CLS names a trainable class token, initially one vector of width D. Expand this same learned vector across the batch. Concatenate it before the final particle states and normalize all rows. Only the CLS row supplies a query. The class token and all real particles supply keys and values.",
        "Report section 11.2; ClassAttentionBlock.forward"),
    "pool_example": (
        "Class attention forms a weighted message, not a mean.",
        "Use an illustrative single head with three available values. Let the class token value be zero and two, and particle values be two and zero, then four and two. Suppose the attention weights are one quarter, one half, one quarter. Multiply each value by its weight, then add: the message is two and one.",
        "Derived worked example of report section 11.2"),
    "pool_update": (
        "Update only CLS through two ordinary attention blocks.",
        "Add the pooled message to the class token, then add its normalized feed forward update. Repeat with the second class block, keeping the final particle table fixed. These class blocks use ordinary multihead attention. They do not receive the relation gates or the pair bias used by the earlier particle blocks.",
        "Report section 11.2"),
    "mask": (
        "One mask convention cannot serve every attention interface.",
        "Our data mask uses true to mean a real particle. Functional scaled dot product attention also treats true as allowed. The class block uses MultiheadAttention, whose key padding mask instead uses true to ignore. Prepend a true keep entry for CLS, then invert the complete keep mask at that interface.",
        "ClassAttentionBlock.forward; GatedDualStreamAttention.forward"),
    "classifier": (
        "The classifier returns logits, not normalized probabilities.",
        "Normalize the final CLS vector. A linear layer maps one hundred twenty eight channels to one hundred twenty eight, followed by GELU and dropout. A final linear layer produces ten logits. Cross entropy expects these raw scores. For evaluation probabilities, apply softmax once. Applying it before cross entropy changes the loss.",
        "Report section 11.3; GDSParT.forward"),
    "mean": (
        "The older mean pooling path is genuinely different.",
        "Version one uses six particle blocks and no class blocks. Normalize the final particle rows, multiply by the real particle mask, sum over particles, and divide by the number of real particles. It does not create an untrained substitute CLS token. Shared sigmoid gates and no pair bias are further configuration differences.",
        "Report sections 11.2, 14.4, 16.2"),
    "equivariance": (
        "Reordering particles must also reorder both relation axes.",
        "Let P represent a permutation. Reordering the particle rows sends X to P X. Reorder the mask in the same way, and send each relation channel to P R P transpose. Then the score matrix becomes P S P transpose. Softmax reorders rows and columns consistently, so particle outputs are equivariant.",
        "Report section 11.4"),
    "invariance": (
        "Pooling removes order without discarding learned content.",
        "Equivariant means the output particle rows follow the input permutation. Invariant means the final jet result stays unchanged. The CLS query attends to the same set of keys and values after reordering, so its weighted sum is unchanged. Test in evaluation mode: independent dropout draws can otherwise hide this property.",
        "Report section 11.4"),
    "code": (
        "Run the residual and pooling arithmetic.",
        "This code executes the two numerical examples we just calculated. Tensor addition gives the residual result. The at operator multiplies the weight row by the value matrix, summing over tokens. Compare the printed results with the animation rather than trusting either in isolation. These are arithmetic checks, not training results.",
        "Executed chapter_13_arithmetic example"),
    "exercise1": (
        "Exercise one: identify the correct final tensor shape.",
        "A batch contains two jets padded to five slots. The embedding width is one hundred twenty eight and there are ten classes. What shape leaves the particle stack, and what shape leaves the classifier? The answers are two by five by one hundred twenty eight, then two by ten. Pooling removes the particle axis.",
        "Solved shape exercise; report section 11"),
    "exercise2": (
        "Exercise two: repair an incomplete particle permutation.",
        "A student permutes the particle table and the mask but only the first relation axis. Is invariance guaranteed? No. A relation still points to an old neighbor address on its second axis. Apply the same permutation to both particle axes of U or R. Only then does the mathematical symmetry argument apply.",
        "Solved permutation exercise; report section 11.4"),
    "exercise3": (
        "Exercise three: distinguish pooling masks and pooled messages.",
        "There is one CLS token, two real particles, and one padded slot. Which entries are ignored by MultiheadAttention? The ignore mask is false, false, false, true. If CLS begins at one and one, and its attention message is two and one, the first residual result is three and two, before the feed forward update.",
        "Solved CLS exercise; report section 11.2"),
    "summary": (
        "Particle blocks update X while reusing R.",
        "The complete forward pass now has no missing junctions. Particle blocks update blue states while reusing orange relations. Class attention converts a set of final particles into one learned jet summary. The classifier returns logits. Consistent permutation of particles, both relation axes, and masks leaves the evaluation result unchanged.",
        "Report section 11"),
}

SUMMARY_13 = [
    "Particle blocks update X while reusing R.",
    "Class attention pools; the classifier returns logits.",
    "Consistent reordering preserves the jet prediction.",
]

CHAPTER_15 = {
    "recap": (
        "A working model is not yet a scientific result.",
        "We can now reconstruct the forward pass, check shapes, count parameters, and differentiate a loss. These checks tell us the implementation behaves as intended. They do not establish that its architecture is better than another model. This final chapter separates training, measurement, comparison, and interpretation.",
        "Report sections 12--18"),
    "objective": (
        "Read results without claiming more than the experiment shows.",
        "We will follow the reported training workflow, calculate metrics, and examine the actual results. Then we will distinguish retraining ablations from gate knockout in a fixed network. Finally, we will cover every limitation in section seventeen and turn those limitations into concrete next experiments.",
        "Report sections 12--18"),
    "workflow": (
        "Separate training, validation selection, and final testing.",
        "Use the training split to update parameters. Use validation to choose the checkpoint. Only then evaluate the chosen checkpoint on the separate test subset. The reported run used ten million training jets, one million validation jets, and one million test jets. Each test class contributed one hundred thousand examples.",
        "Report sections 12, 14.1"),
    "loss": (
        "Smoothed targets distribute a small mass across all classes.",
        "There are ten classes. With smoothing epsilon equal to zero point one, the true class target is zero point nine one. Each other class receives zero point zero one. These add to one. Compute cross entropy from logits; PyTorch handles the stable log softmax internally. Smoothing changes the training target, not the recorded class label.",
        "Report section 12.1"),
    "step": (
        "Accumulate gradients before clipping and stepping.",
        "Set gradients to zero, run the model, calculate loss, backpropagate, clip the global gradient norm, and take one AdamW step. Update the moving average after the optimizer step. For unequal micro batches, weight each mean loss by its fraction of the full batch. Do not divide every piece by the number of pieces.",
        "Report sections 12.2--12.4"),
    "adamw": (
        "AdamW combines adaptive gradients with a separate weight decay.",
        "Consider one weight initially equal to one, with gradient two and zero initial moments. The first corrected moments are two and four. Their ratio is approximately one, after dividing by the square root of four. With learning rate one thousandth and weight decay zero point zero one, the new weight is approximately zero point nine nine eight nine nine.",
        "Report section 12.2; derived scalar first-step example"),
    "microbatch": (
        "Unequal micro batches require unequal loss weights.",
        "Suppose a batch of five jets is split into pieces of three and two. The mean losses are two and five. The full batch mean is three fifths times two plus two fifths times five, which equals three point two. Averaging the two means gives three point five and is wrong. Apply the same size fractions before accumulating gradients.",
        "Report section 12.4; derived micro-batch example"),
    "recipe": (
        "The reported recipe is part of the evidence.",
        "The reported shared model trained for ten epochs with batch size five hundred twelve, seed zero, and AdamW peak learning rate one thousandth. Weight decay was zero point zero one. Warm up covered five percent of steps, followed by cosine decay. Gradient clipping used norm one. Dropout and label smoothing were both zero point one.",
        "Report section 12.5"),
    "ema": (
        "A moving average can lag behind a short run.",
        "The reported run used moving average decay zero point nine nine nine nine over one hundred ninety five thousand three hundred twenty steps. The effective averaging horizon is about ten thousand steps. Short ablations used only one thousand one hundred seventy two steps. Decay zero point nine nine nine can then retain substantial influence from early weights.",
        "Report sections 12.3, 12.5, 15.1"),
    "memory": (
        "Saving activations differs from saving a resumable checkpoint.",
        "CUDA mixed precision and gradient scaling reduce practical training cost. Activation checkpointing recomputes block internals during backward instead of retaining every intermediate. A disk checkpoint is different: it saves weights, optimizer state, schedule, moving average state, and progress. Neither method changes the quadratic number of pair relationships.",
        "Report sections 12.4, 16.4"),
    "accuracy": (
        "Accuracy measures how often the largest score is correct.",
        "For a small illustrative batch, predictions are zero, two, one, one, while true labels are zero, one, one, one. Three of four predictions are correct, so accuracy is zero point seven five. A confusion matrix counts true class by predicted class and reveals which mistakes the average conceals.",
        "Report section 13; derived four-example calculation"),
    "auc": (
        "AUC measures ranking, not the fraction correctly classified.",
        "For two classes, compare a signal score with a background score. A correctly ordered pair earns one, and a tie earns one half. With signal scores zero point nine and zero point six, and background scores zero point eight and zero point two, three of four pairs are ordered correctly. The illustrative AUC is zero point seven five.",
        "Report section 13; derived pair-ranking calculation"),
    "macro": (
        "The reported AUC averages all forty five class pairs.",
        "The report uses macro one versus one AUC, not one versus rest. Ten classes form forty five unordered pairs. For each pair, average the two directional score comparisons, then average equally across the pairs. Do not compare this value with a metric using a different averaging definition and call the difference architectural.",
        "Report section 13; compute_jet_tagging_metrics"),
    "rejection": (
        "Large rejection values can depend on very few jets.",
        "At a chosen signal efficiency, background rejection is one divided by the background acceptance rate. If ten of one hundred thousand background jets survive, acceptance is one in ten thousand, so rejection is ten thousand. Only ten survivors determine that estimate. Its rough relative counting uncertainty is one over the square root of ten, about thirty two percent.",
        "Report section 13"),
    "working_point": (
        "Specify the signal efficiency before quoting background rejection.",
        "For a signal class versus QCD, use the signal probability divided by signal plus QCD probability, with a tiny stabilizing constant. Sweep a threshold on this score. The report uses fifty percent signal efficiency for most signals, ninety nine percent for the Higgs semileptonic class, and ninety nine point five percent for the semileptonic top class.",
        "Report section 13; compute_jet_tagging_metrics"),
    "uncertainty": (
        "Test counting error differs from training seed variation.",
        "For one million independent test examples, a rough binomial standard error is the square root of accuracy times one minus accuracy, divided by the test count. At accuracy zero point eight four eight four, it is about zero point zero zero zero three six. This describes test counting variability, not the unknown variation from retraining with different seeds.",
        "Report section 13; distinction from section 17 item 1"),
    "code": (
        "Execute the metric calculations on illustrative inputs.",
        "The displayed program calculates accuracy from equality of predicted and true labels. It calculates binary AUC by enumerating all four signal background comparisons, including half credit for ties. Finally it divides the background count by the survivor count. These outputs verify arithmetic; they are not the measured JetClass results.",
        "Executed chapter_15_metrics example"),
    "identities": (
        "Keep four different model identities separate.",
        "Published ParT is the paper's model and training recipe. The local bias only proxy is a GDS implementation with gates disabled, not an official ParT reconstruction. The reported hybrid has shared gates and additive bias. The local default instead uses conditioned gates in each block. A default constructor is not evidence that the default achieved the reported result.",
        "Report sections 1.4, 9.5, 16.2"),
    "results": (
        "The hybrid approaches, but does not exceed, published ParT.",
        "The ten epoch shared hybrid achieved accuracy zero point eight four eight four and macro AUC zero point nine eight five three. Published ParT trained on ten million unique jets reports zero point eight five zero and zero point nine eight six zero. Version one achieved zero point eight two seven two and zero point nine eight two four.",
        "Report section 14.3"),
    "uncontrolled": (
        "Equal unique training size does not mean equal training exposure.",
        "The shared run saw one hundred million jet presentations: ten million jets times ten epochs. Published ParT saw about five hundred twelve million presentations and used Lookahead with RAdam rather than AdamW. The test sizes were also different: one million versus twenty million. These are important uncontrolled differences, not evidence that either architecture is intrinsically superior.",
        "Report section 14.3"),
    "validation": (
        "A reference checkpoint checks the pipeline, not training fairness.",
        "The official pretrained ParT checkpoint, trained on one hundred million jets, was evaluated through the local pipeline on two hundred thousand test jets. Accuracy zero point eight six one two and AUC zero point nine eight seven seven one agree closely with published reference values. This supports preprocessing and metric consistency, but does not control the separate training experiments.",
        "Report section 14.2"),
    "v1": (
        "Several simultaneous changes prevent a single cause explanation.",
        "Version two improves accuracy over version one by about two point one percentage points. But gate activation, pair bias, block count, pooling, label smoothing, moving average, and training epochs all changed. We cannot assign that entire gain to gates, class attention, or any other single change. A controlled ablation changes one factor at a time.",
        "Report section 14.4"),
    "ablation": (
        "The small retraining difference remains unresolved.",
        "Without moving average evaluation, shared gates plus bias reached zero point six nine six five eight eight, and bias only reached zero point six nine four two four two. Their difference is about zero point two three five percentage points. The budget was three hundred thousand jets and two epochs, with one seed and no measured error bars.",
        "Report section 15.1"),
    "ablation_scope": (
        "Do not replace missing error bars with confident language.",
        "The conditioned gates plus bias result was zero point six nine three three three zero. Conditioned gates without bias reached zero point six two two two six eight. Gates only were weaker at this short budget. But the small shared versus bias only difference is neither established superiority nor measured noise. Repeated seeds and paired analysis are needed.",
        "Report section 15.1"),
    "knockout": (
        "Removing gates tests reliance inside an already fitted network.",
        "On fifty thousand validation jets, keeping both trained gates gave zero point eight four eight two accuracy. Replacing both gates by ones gave zero point seven five two one. The fitted network therefore relies on its learned gates. This is not a comparison with a new model trained from scratch without gates: its other weights could adapt differently.",
        "Report section 15.2; exploratory diagnostic"),
    "interaction": (
        "Knockout effects need not add independently.",
        "Replacing only the key gate gave zero point eight three zero two. Replacing only the query gate gave zero point seven two three two. Removing both was not the sum of those two changes. Co adapted components interact. These exploratory values came from one off session commands, not a committed reproducible analysis script in the report.",
        "Report section 15.2"),
    "correlation": (
        "Gate averages cannot establish the direction of information flow.",
        "The report describes correlations between average gates and particle kinematics. But a gate average hides channel signs and query key alignment. It cannot prove that core particles attend to peripheral particles. The shared gate directly reads pair kinematics, not particle identification. Measure actual attention changes, control correlated variables, and remember that values and later layers also affect decisions.",
        "Report section 15.3"),
    "limits_evidence": (
        "Limitations one through three concern experimental control.",
        "First, results use one seed with no seed variation error bars. Second, the ablation training budget is small and may rank models differently from full training. Third, there is no controlled full budget ParT baseline. The local bias only experiment is a useful proxy, but not a replacement for that missing comparison.",
        "Report section 17, items 1--3"),
    "limits_claims": (
        "Limitations four through six restrict the interpretation.",
        "Fourth, the reported model is a hybrid: it retains additive bias, so it does not demonstrate that gates replace bias. Fifth, high rejection estimates have limited test subset statistics. Sixth, knockout and correlation diagnostics are exploratory and need a committed reproducible analysis. These are boundaries on our conclusions, not reasons to hide the measurements.",
        "Report section 17, items 4--6"),
    "limits_structure": (
        "Limitations seven through nine concern architecture and provenance.",
        "Seventh, relations are static and shared gates are reused; the conditioned variant was not trained at full budget in the report. Eighth, pair memory scales quadratically with particle count. Ninth, generated results prose retained a stale version one caveat. The technical report corrects that prose: recorded configuration and code include class attention and additive bias in shared version two.",
        "Report section 17, items 7--9; results.json and gds_part_colab.py authoritative"),
    "memory_example": (
        "One million relation entries arise from one small jet.",
        "With one hundred twenty eight particle slots and relation width sixty four, R contains one million forty eight thousand five hundred seventy six numbers per jet. Float thirty two requires four bytes per number: four mebibytes. A batch of five hundred twelve needs two gibibytes for R alone, before gradients and other activations. Fewer weights do not guarantee low activation memory.",
        "Report section 16.4"),
    "next_control": (
        "Prioritize experiments that resolve the current uncertainty.",
        "Finish and evaluate the twenty epoch run before claiming its final result. Repeat shared gates plus bias versus bias only at full budget with at least three seeds, alongside a same pipeline ParT comparison. Train the conditioned gate variant at full budget. Evaluate the full twenty million test jets to better resolve rare background survivors.",
        "Report section 18, items 1--3 and 6"),
    "next_models": (
        "Extensions remain proposals, not demonstrated improvements.",
        "Commit controlled knockout and attention change analyses. Possible extensions include gating values, updating relation states between blocks, and giving gates direct pair access. The report also points to a per particle mixture of experts extension. Each proposal changes a testable hypothesis; none is a result until implemented, trained, and evaluated with appropriate controls.",
        "Report section 18, items 4, 5, 7"),
    "exercise1": (
        "Exercise one: calculate and interpret a small improvement.",
        "Subtract zero point six nine four two four two from zero point six nine six five eight eight. The answer is zero point zero zero two three four six, or zero point two three four six percentage points. Can we call this reproducible superiority? No. We have only one seed at a short training budget and no uncertainty estimate.",
        "Solved exercise using report section 15.1"),
    "exercise2": (
        "Exercise two: distinguish knockout from retraining.",
        "A trained hybrid loses accuracy when both gates become one. Which conclusion follows: the fitted model uses its gates, or every model trained without gates must be worse? Only the first follows. To test the second claim, retrain a bias only control under a matched budget and repeat across seeds.",
        "Solved causal interpretation exercise; report section 15.2"),
    "exercise3": (
        "Exercise three: calculate the surviving background count.",
        "A test contains one hundred thousand background jets and reports rejection ten thousand. How many background jets survive approximately? Divide the count by rejection to obtain ten. The rough relative counting uncertainty is about thirty two percent. More test events improve this estimate; extra decimal places do not create extra information.",
        "Solved statistics exercise; report section 13"),
    "summary": (
        "Implementation checks establish correctness, not superiority.",
        "You can now follow the mathematics into an independent implementation and inspect its training evidence. The reported shared hybrid is promising, but its comparison with published ParT is uncontrolled. Gate knockout shows reliance in one fitted network. The next reliable progress comes from reproducible, matched experiments with uncertainty estimates.",
        "Report sections 14--18"),
}

SUMMARY_15 = [
    "Implementation checks establish correctness, not superiority.",
    "Knockout measures reliance; retraining tests architecture choices.",
    "Controlled experiments support stronger conclusions.",
]

CHAPTER_13_CODE = {
    "arithmetic": """x = torch.tensor([2., -1.])
a = torch.tensor([.5, 1.])
f = torch.tensor([-.5, 2.])
print((x + a + f).tolist())
w = torch.tensor([.25, .5, .25])
v = torch.tensor([[0., 2.],
                  [2., 0.], [4., 2.]])
print((w @ v).tolist())""",
}

CHAPTER_15_CODE = {
    "metrics": """y = torch.tensor([0, 1, 1, 1])
p = torch.tensor([0, 2, 1, 1])
print((p == y).float().mean().item())
s = torch.tensor([.9, .6])[:, None]
b = torch.tensor([.8, .2])[None, :]
auc = ((s > b) + .5 * (s == b)).mean()
print(auc.item())
print(100000 // 10)""",
}


def _scene_steps(filename):
    """Audit the same literal equations that custom scenes actually render."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(__file__).with_name(filename).read_text(encoding="utf-8"))
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        formulas, teaching_keys = [], []
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            name = call.func.attr
            if name in {"equations", "diagram"} and len(call.args) > 1:
                try:
                    result[ast.literal_eval(call.args[0])] = tuple(ast.literal_eval(call.args[1]))
                except (ValueError, TypeError):
                    pass
            if name == "formula" and call.args:
                try:
                    formulas.append(ast.literal_eval(call.args[0]))
                except (ValueError, TypeError):
                    pass
            if name == "teach" and call.args:
                try:
                    teaching_keys.append(ast.literal_eval(call.args[0]))
                except (ValueError, TypeError):
                    pass
        for key in teaching_keys:
            labels = [item.value for item in ast.walk(node)
                      if isinstance(item, ast.Constant) and isinstance(item.value, str)
                      and r"\text{" in item.value]
            result[key] = tuple(dict.fromkeys(formulas + labels))
    return result


from course_content import b

CODE = {
    "c13_arithmetic": "import torch\n" + CHAPTER_13_CODE["arithmetic"],
    "c15_metrics": "import torch\n" + CHAPTER_15_CODE["metrics"],
}
CHAPTERS = {}
for _number, _chapter, _filename in (
    (13, CHAPTER_13, "chapter_13_complete_forward_pass.py"),
    (15, CHAPTER_15, "chapter_15_experiments_limitations.py"),
):
    _steps = _scene_steps(_filename)
    CHAPTERS[_number] = {
        key: b(caption, narration, *_steps.get(key, ()), source=source)
        for key, (caption, narration, source) in _chapter.items()
    }
CHAPTERS[13]["code"] = b(*CHAPTER_13["code"][:2], "c13_arithmetic",
                         kind="code", source=CHAPTER_13["code"][2])
CHAPTERS[15]["code"] = b(*CHAPTER_15["code"][:2], "c15_metrics",
                         kind="code", source=CHAPTER_15["code"][2])
CHAPTERS[13]["summary"].steps = tuple(r"\text{" + line + "}" for line in SUMMARY_13[1:])
CHAPTERS[15]["summary"].steps = tuple(r"\text{" + line + "}" for line in SUMMARY_15[1:])

# These are the companion author's independently rebuilt modules, not production
# imports. Tests are sequential so named tensors visibly survive each step.
from companion.chapter_examples import EXAMPLES

CODE_PRELUDES = {14: EXAMPLES["c14_imports"]}
CODE.update({key: value for key, value in EXAMPLES.items()
             if key not in {"c14_imports", "c14_counts", "c14_breakdown"}})
CODE.update({
    "c14_counts_a": """counts = parameter_audit()
for name in ("v1", "shared"):
    print(name, counts[name])""",
    "c14_counts_b": """for name in ("conditioned", "no_gates"):
    print(name, counts[name])""",
    "c14_audit_a": """parts = list(shared_breakdown().values())
print(*parts[:4], sep=" + ")
print("subtotal", sum(parts[:4]))""",
    "c14_audit_b": """print(*parts[4:7], sep=" + ")
print("subtotal", sum(parts[4:7]))""",
    "c14_audit_c": """print(*parts[7:], sep=" + ")
print("total", sum(parts))""",
    "c14_permutation": """order = torch.tensor([2, 0, 3, 1])
other = model(x[:, order],
    u[:, order][:, :, order], keep[:, order])
print(torch.allclose(logits, other, atol=1e-6))
print(tuple(other.shape))""",
    "c14_padding": """changed = x.clone()
changed[~keep] = 1000
again = model(changed, u, keep)
print(torch.allclose(logits, again, atol=1e-6))""",
})

CHAPTERS[14] = {
    "recap": b("Rebuild the operations instead of importing the original.",
        "The complete forward pass now connects both embeddings, gates, particle blocks, pooling, and classification. We will rebuild that chain in a readable companion module. It does not import the production model. Production code is used only in separate equivalence tests, where an independent implementation is precisely what we want to compare.",
        r"F,U,m\longrightarrow X,R\longrightarrow c\longrightarrow\ell",
        source="Companion gds_part_reconstruction.py; report sections 11, 16"),
    "objective": b("Build one module, inspect its output, then connect the next.",
        "Each implementation excerpt comes from the independent companion. We will follow it with an executed shape or behavior check. Use a tiny width while debugging so the tensors are easy to inspect. Afterwards instantiate the real configurations and count their actual parameters. Small synthetic examples verify mechanics, not physics performance.",
        r"D=16,\quad D_p=8,\quad H=2,\quad L=2",
        r"\text{debug first}\longrightarrow\text{full configuration audit}", kind="flow",
        source="Companion chapter_examples.py"),
    "inputs": b("Begin with separate particle, pair, and keep tensors.",
        "The seed is fixed at seventeen so these demonstrations are repeatable. Create two jets with four slots and seventeen features per slot. Each pair has four inputs. The first jet has three real particles, while the second has two. The pair axis is not the head axis, even if a small example happens to use matching sizes.",
        "c14_inputs", kind="code", source="Executed companion chapter_examples.py"),
    "build_particle": b("Write the particle embedding as a shared channel transformation.",
        "The embedding first normalizes seventeen features, then applies a linear map, GELU, another linear map, and dropout. PyTorch applies these operations to the last axis of every particle row. The same learned weights serve every row. The forward method returns the sequential network's result without flattening the batch or particle axes.",
        source="Independent ParticleEmbedding implementation"),
    "particle": b("Check that particle embedding preserves both leading axes.",
        "Instantiate the particle embedding with output width sixteen and dropout zero for this check. Execute it on the two by four by seventeen input. The output is two by four by sixteen. This confirms that only the feature width changed. It does not by itself prove numerical correctness, which is why the companion also tests equivalent operations.",
        "c14_particle", kind="code", source="Executed companion chapter_examples.py"),
    "build_pair": b("Write the pair embedding without mixing neighbor addresses.",
        "The pair embedding has two linear maps separated by GELU. It does not include the particle embedding's initial LayerNorm or its dropout. Each ordered pair receives the same four to eight to eight transformation in this tiny example. Preserve both particle axes, because later masking and aggregation must still identify the neighbor index.",
        source="Independent PairwiseEmbedding implementation"),
    "pair": b("Verify the relation tensor before constructing any gates.",
        "Run the pair embedding on the raw pair tensor. Its shape becomes two by four by four by eight. The final eight is the relation width, not the number of attention heads. Keep this relation tensor alive: both gate construction and additive bias will read it, and the full model computes its embedding only once.",
        "c14_pair", kind="code", source="Executed companion chapter_examples.py"),
    "build_shared": b("Aggregate valid neighbors before constructing two separate gates.",
        "Insert singleton axes into the keep mask so it aligns with the neighbor axis and relation channels. Multiply relations by that mask, sum over neighbors, and divide by their count. A diagonal self relation remains included when its particle is real. Two separate gate networks map the summary into query and key channels.",
        source="Independent RelationalGate.aggregate and forward"),
    "build_identity": b("Zero the final affine layers, not every gate parameter.",
        "The identity gate is one plus hyperbolic tangent. Zero initialization of each final affine layer gives a zero preactivation, so every initial gate is exactly one. Earlier layers remain normally initialized. Their gradient can initially be blocked by the zero final weights; the final layers can still learn and open that path during subsequent steps.",
        source="Independent RelationalGate.make_mlp; report sections 7, 10"),
    "shared": b("Test both the gate shape and the identity initialization.",
        "Execute the shared gate on the relation tensor and keep mask. Both outputs have shape two by four by sixteen, matching the particle state. Compare every entry with one. Both equality checks are true. A sigmoid initialized at zero would instead produce one half, so this test distinguishes the two gate conventions.",
        "c14_shared_gate", kind="code", source="Executed companion chapter_examples.py"),
    "build_conditioned": b("Conditioned summaries read the current normalized particle state.",
        "Project the current state from particle width into relation width. For every query particle, compare its projected context with each neighbor relation, divide by the square root of relation width, mask invalid neighbors, and normalize with softmax. The weighted relation sum replaces the uniform mean. This does not update the stored relation tensor.",
        source="Independent RelationalGate.aggregate"),
    "conditioned": b("The context projection adds parameters and changes the summary.",
        "Construct the conditioned gate, normalize the current states, and execute it. The output shape remains two by four by sixteen. The context weight has shape eight by sixteen. Initial identity outputs can still all equal one, even though the summary calculation differs. Therefore equal initial gates do not mean the two architectures are identical.",
        "c14_conditioned", kind="code", source="Executed companion chapter_examples.py"),
    "bias": b("Project relations into heads, then move the head axis.",
        "A linear map converts eight relation channels to two head bias channels. Its immediate output has batch, query, key, then head axes. Permute it into batch, head, query, key order. The printed shape is two by two by four by four. This is additive bias; the red gates are a separate mechanism.",
        "c14_bias", kind="code", color="B", source="Executed companion chapter_examples.py"),
    "build_attention": b("Implement the score calculation explicitly before optimizing it.",
        "Project query, key, and value vectors. Multiply only query and key by their respective gates. Split the particle width into heads, then multiply query by transposed key and divide by the square root of head width. The companion writes this multiplication explicitly so every operation can be inspected; production uses an optimized attention operation.",
        source="Independent GatedAttention.forward; production GatedDualStreamAttention.forward"),
    "build_attention_mix": b("Mask scores before softmax, then mix values.",
        "Add pair bias and exclude padded key columns with negative infinity. Softmax runs along the key axis. During training, dropout acts on attention weights before value mixing. Concatenate the heads and apply the output projection. Padded query rows may exist, but excluded keys cannot pass their content into real particle queries.",
        source="Independent GatedAttention.forward"),
    "attention": b("Inspect attention shape and one masked probability.",
        "Execute the attention module with the current states, gates, pair bias, and keep mask. The mixed state is two by four by sixteen. Attention weights are two by two by four by four. The fourth key of the first jet is padding, and its displayed probability is exactly zero. Finite padding still needs correct masking everywhere.",
        "c14_attention", kind="code", source="Executed companion chapter_examples.py"),
    "build_block": b("Write both residual paths in their correct order.",
        "Normalize the incoming state, calculate conditioned gates if this block owns them, and add the attention message to the incoming state. Then normalize that updated state and add its feed forward output. Do not feed the original state into the second normalization. This compact forward method is the recurrence from the previous chapter.",
        source="Independent GDSParTBlock.forward"),
    "block": b("A complete block must preserve the state shape.",
        "Create a tiny configuration with two particle blocks and one class block. Its feed forward hidden width is thirty two. Execute one particle block. The output remains two by four by sixteen. Residual addition requires matching dimensions, but preserving shape alone does not ensure the relation mask or gate mode is correct.",
        "c14_block", kind="code", source="Executed companion chapter_examples.py"),
    "build_cls": b("Use one query while preserving all valid keys and values.",
        "Concatenate CLS before the particles and normalize the combined rows. Build a valid mask with CLS always retained, then invert it for MultiheadAttention's ignore convention. Query with only the first normalized row. Add the returned message to CLS, then its normalized feed forward update. Particle states are not rewritten in this class block.",
        source="Independent ClassAttentionBlock.forward"),
    "cls": b("Test that pooling returns exactly one token per jet.",
        "For this module check, start CLS at zero with shape two by one by sixteen. The complete model instead owns a learned initialized token. The class block returns the same CLS shape. It does not return a newly updated four particle table. This is attention pooling, not an ordinary self attention layer over every row.",
        "c14_cls", kind="code", source="Executed companion chapter_examples.py"),
    "build_model": b("Construct modules once and register repeated blocks correctly.",
        "The full model owns the two embeddings, optional shared gates, and optional pair bias. Store particle blocks and class blocks in ModuleList so PyTorch discovers their parameters. Store the class token as a Parameter, not an ordinary tensor. The classifier has a hidden linear layer, GELU, dropout, and ten output channels.",
        source="Independent GDSParT.__init__"),
    "build_forward": b("The full forward pass reuses relations and returns raw logits.",
        "Embed particles and pairs, prepare bias and shared gates, then walk through particle blocks while carrying the same relation tensor. Next choose class attention or masked mean pooling from the configuration. Normalize the appropriate pooled representation and apply the classifier. The optional trace records intermediate shapes without changing the predicted logits.",
        source="Independent GDSParT.forward"),
    "logits": b("Run the assembled model and inspect its execution trace.",
        "The model is in evaluation mode, so dropout is disabled. Executing the full tiny model produces two by ten logits. The trace reports one relation embedding evaluation and two particle block output shapes. This verifies the actual path being run, rather than relying only on a diagram saying that R is reused.",
        "c14_logits", kind="code", source="Executed companion chapter_examples.py"),
    "backward": b("A differentiable model must send finite gradients to its classifier.",
        "Use the two illustrative labels to compute cross entropy from the raw logits. Backpropagate once, then inspect the final classifier weight gradient. Every entry is finite, and the total absolute gradient is nonzero. This is a useful diagnostic, not proof that all earlier parameters must receive nonzero gradients at identity gate initialization.",
        "c14_backward", kind="code", source="Executed companion chapter_examples.py"),
    "permutation": b("Test the symmetry using both pair axes.",
        "Reorder the four slots, applying the same permutation to particle rows, both pair axes, and the keep mask. Execute the same model again. The logits agree within floating point tolerance. We deliberately compare in evaluation mode. The test catches many axis mistakes that ordinary shape checks would miss.",
        "c14_permutation", kind="code", source="Executed reconstruction permutation test"),
    "padding": b("Change padded particle features and check the real prediction.",
        "Replace padded particle feature rows with a large finite number. Keep the same real particle mask and rerun the model. The output should agree, because padding cannot contribute as keys or as pooled content. This test does not promise that arbitrary NaNs are harmless: multiplying NaN by zero does not repair it.",
        "c14_padding", kind="code", source="Executed reconstruction padding test; cache masking caveat"),
    "counts_a": b("Count parameters on the live full configurations.",
        "The audit constructs real full size models and sums the number of elements in their registered parameters. Version one has one million two hundred eighty thousand six hundred eighty four. Reported shared version two has two million seventy four thousand four hundred twenty. Version one requires six blocks as well as sigmoid gates, no pair bias, and mean pooling.",
        "c14_counts_a", kind="code", source="Live companion parameter_audit"),
    "counts_b": b("The conditioned model exceeds the reported shared count.",
        "The default conditioned model has two million four hundred eighty eight thousand one hundred sixteen parameters. The no gates bias only model has two million twenty four thousand seven hundred fifty six. The shared gate therefore adds forty nine thousand six hundred sixty four parameters. Never label the default conditioned count as the reported shared count.",
        "c14_counts_b", kind="code", source="Live companion parameter_audit"),
    "audit_a": b("Audit embeddings, shared gates, and pair bias separately.",
        "The particle embedding contributes eighteen thousand eight hundred fifty parameters. The pair embedding contributes four thousand four hundred eighty. The shared query and key gates contribute forty nine thousand six hundred sixty four. The pair bias projection contributes five hundred twenty. The displayed program reads these numbers from the actual model, not from rounded labels.",
        "c14_audit_a", kind="code", source="Live companion shared_breakdown"),
    "audit_b": b("Count every repeated block without duplicating shared modules.",
        "Across eight particle blocks, attention projections contribute five hundred twenty eight thousand three hundred eighty four. The two LayerNorm modules per block contribute four thousand ninety six. Feed forward networks contribute one million fifty three thousand six hundred ninety six. Multiply genuinely separate block parameters by eight, but do not multiply the shared relation embedding or shared gate.",
        "c14_audit_b", kind="code", source="Live companion shared_breakdown"),
    "audit_c": b("Complete the total with pooling, normalization, and classification.",
        "The learned CLS token adds one hundred twenty eight parameters. Two class attention blocks add three hundred ninety six thousand five hundred forty four. Final normalization adds two hundred fifty six, and the classifier adds seventeen thousand eight hundred two. The final sum is exactly two million seventy four thousand four hundred twenty.",
        "c14_audit_c", kind="code", source="Live companion shared_breakdown"),
    "debugging": b("Debug contracts before chasing model quality.",
        "First check input axes and Boolean mask conventions. Then verify finite tensors and attention row sums, confirm identity gates, and compare permutation and finite padding tests. Reject an entirely empty jet rather than depending on all masked softmax behavior. Finally compare copied parameters against production in separate tests. A passed synthetic overfit check is not a JetClass benchmark.",
        r"\text{shapes}\to\text{masks}\to\text{finite values}",
        r"\text{identity}\to\text{permutation}\to\text{padding}",
        r"\text{independent rebuild}\ \leftrightarrow\ \text{production test oracle}",
        kind="flow", source="Companion tests.py and train_demo.py"),
    "exercise1": b("Exercise one: derive the pair bias parameter count.",
        "The full relation width is sixty four and there are eight attention heads. A linear layer stores an eight by sixty four weight matrix and eight biases. Multiply to obtain five hundred twelve weights, then add eight to obtain five hundred twenty. The number of particle pairs changes activation size, not this learned parameter count.",
        r"W_{\rm bias}:(8,64),\quad b_{\rm bias}:(8)",
        r"8\cdot64+8=512+8=520",
        source="Solved parameter exercise; report section 16.3"),
    "exercise2": b("Exercise two: explain the shared versus conditioned count difference.",
        "One shared gate has forty nine thousand six hundred sixty four parameters. A conditioned gate additionally projects one hundred twenty eight channels to sixty four, adding eight thousand two hundred fifty six. Eight conditioned gates replace one shared gate. The increase is eight times fifty seven thousand nine hundred twenty, minus forty nine thousand six hundred sixty four: four hundred thirteen thousand six hundred ninety six.",
        r"128\cdot64+64=8256",
        r"8(49664+8256)-49664=413696",
        r"2074420+413696=2488116",
        source="Solved architecture audit; live counts"),
    "exercise3": b("Exercise three: find a test that exposes the axis bug.",
        "A programmer transposes a pair axis with a head axis, but both happen to have size four in the debugging batch. Does a matching output shape establish correctness? No. Use unequal axis sizes and run a consistent particle permutation test. Also inspect which key columns receive zero probability under the real particle mask.",
        r"N=5,\quad H=2\quad\text{exposes equal-size axis accidents}",
        r"\ell(PF,PUP^\top,Pm)\stackrel{?}{=}\ell(F,U,m)",
        source="Solved debugging exercise"),
    "summary": b("An independent reconstruction exposes every tensor contract.",
        "The companion now contains the complete independent model, not a wrapper around production. We examined implementation excerpts and ran each module with visible outputs, then tested the complete prediction and exact parameter counts. Keep those tests close to the code: a model name without configuration, masks, and reproducible checks is not a sufficient specification.",
        r"\text{modules}\longrightarrow\text{executed tests}\longrightarrow\text{full model}",
        r"\text{shared}=2074420,\quad\text{conditioned}=2488116",
        source="Companion reconstruction, examples, tests, and parameter audit"),
}


def _insert_before(chapter, anchor, key, beat):
    entries = {}
    for existing, item in chapter.items():
        if existing == anchor:
            entries[key] = beat
        entries[existing] = item
    return entries


CHAPTERS[14] = _insert_before(CHAPTERS[14], "build_attention_mix", "build_attention_mask",
    b("Preserve the padding rule when adding pair bias.",
      "A floating point attention mask can carry both learned bias and padding exclusions. Convert the bias to the query datatype, replace invalid key columns by negative infinity, and add it to the scores. If bias is absent, apply the exclusion directly to the scores. Never apply the mask after softmax and assume the remaining weights still sum to one.",
      source="Independent GatedAttention.forward; production mask semantics"))
CHAPTERS[14] = _insert_before(CHAPTERS[14], "build_model", "build_model_embed",
    b("Own shared modules once, and register particle blocks individually.",
      "The constructor creates one particle embedding and one pair embedding. It creates a shared gate only when the configuration requests shared gates, and a bias projection only when enabled. ModuleList registers a separate instance of every particle block. Repeating the same block object would share its weights and produce the wrong parameter count.",
      source="Independent GDSParT.__init__"))
CHAPTERS[14] = _insert_before(CHAPTERS[14], "logits", "build_readout",
    b("Select the readout without changing normalization order.",
      "In the class attention branch, expand the learned token across the batch and update it while particle states remain fixed. Normalize the resulting CLS vector. In the mean branch, normalize each particle first, then compute the masked mean. Finally return logits, or logits with the optional diagnostic trace. Both branches reach the same classifier.",
      source="Independent GDSParT.forward"))
CHAPTERS[14]["summary"].steps = (
    r"\text{Test every module before connecting the full model.}",
    r"\text{Audit parameters using the exact configuration.}",
)

# Custom scene prose is registered even when it is not a beat caption.
EXTRA_SCREEN_TEXT = SUMMARY_13 + SUMMARY_15 + [
    "Test every module before connecting the full model.",
    "Audit parameters using the exact configuration.",
]
EXTRA_FORMULAS = [
    r"Q'=Q\odot G_Q",
    r"K'=K\odot G_K",
    r"S=Q'K'^\top/4+\mathcal B+M",
    r"\binom{0}{2}", r"\binom{2}{0}", r"\binom{4}{2}",
    r"\binom{0}{0.5}", r"\binom{1}{0}", r"\binom{1}{0.5}",
    r"\tfrac14", r"\tfrac12", r"\tfrac14",
]
