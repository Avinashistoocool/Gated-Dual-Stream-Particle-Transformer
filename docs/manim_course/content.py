"""English script and screen-text registry for the first review milestone.

Source references are PDF page numbers, not PDF viewer offsets.
Worked outputs are calculated from the report's illustrative inputs.
"""
COLORS = {
    "X": "#58A6FF", "R": "#FFAA55", "Q": "#67D98B", "K": "#BB88EE",
    "V": "#F4D35E", "G": "#FF6464", "B": "#45C9BD", "M": "#939CAA",
}

# Each entry is (on-screen sentence, spoken explanation, source).
CHAPTER_00 = {
    "recap": (
        "We will turn mathematics into a working particle classifier.",
        "Welcome. We will build a particle classifier from mathematics you can inspect, not from unexplained boxes. You need school arithmetic and curiosity. We will introduce each new operation, calculate an example, and connect it to code. This first chapter establishes our destination.",
        "PDF pp. 6-7"),
    "objective": (
        "You will identify the inputs, outputs, and two streams.",
        "By the end of this chapter, you will identify what enters the model, what leaves it, and why two streams are needed. Understanding every internal operation comes later. For now, follow the moving particles and their relationships rather than memorising the architecture.",
        "PDF pp. 10-12"),
    "jet": (
        "A jet contains particles, but receives one classification.",
        "The report describes a jet as a collimated collection of reconstructed particles. Each dot represents one particle. The enclosing boundary represents one example. The prediction belongs to that entire example, not separately to every dot. A jet therefore enters as a collection.",
        "PDF p. 6"),
    "features": (
        "Each particle contributes seventeen measured or derived features.",
        "The collection first becomes a table. Each row describes a particle, and each column holds one feature. There are seventeen input features in this implementation. Those seventeen observations are not seventeen trainable parameters. Learning adjusts the model's weights, not the recorded observations.",
        "PDF pp. 8-9, 33"),
    "pairs": (
        "Each pair contributes four descriptions of its relationship.",
        "Now select two particles. Their relationship needs two addresses: the first particle and the second particle. The pair input contains four kinematic features. Notice the orange connection becoming an orange table entry. A relation belongs to a pair, rather than to one particle alone.",
        "PDF pp. 9-12"),
    "dimensions": (
        "A shape records the meaning of every array axis.",
        "We need names for sizes before displaying shapes. Capital B counts jets in a batch. N counts padded particle slots. D counts particle channels. D sub p counts relation channels. These letters describe dimensions. They are not additional measurements supplied by the detector.",
        "PDF p. 7"),
    "streams": (
        "Particle states and pair relations form the two streams.",
        "X names the particle states and stays blue. R names the pair representations and stays orange. X has one particle axis. R has two particle axes. In the reported configuration, their channel widths are one hundred twenty eight and sixty four. This is what dual stream means.",
        "PDF pp. 10-12"),
    "reuse": (
        "Particle states change while the relation representation is reused.",
        "Watch the blue states move through successive blocks. The orange relation table stays in place and supplies information to each block. It is computed once per forward pass, not updated by a second recurrent transformer. Dual stream does not mean queries and keys.",
        "PDF p. 12"),
    "hybrid": (
        "The reported model combines gates with additive pair bias.",
        "The orange stream has two routes into particle attention. Red gates rescale query and key channels. Teal pair bias adjusts comparison scores. The reported model keeps both routes, so it is a hybrid. We will derive these routes separately before combining them.",
        "PDF pp. 11-12"),
    "comparison": (
        "The published model and local baseline are different systems.",
        "Keep three comparisons distinct. Original Particle Transformer is the published architecture. Turning gates off here creates a local bias-only proxy, not an exact reproduction. The reported GDS model shares gates across blocks. The code's default instead uses separate, state-conditioned gates within each block.",
        "PDF pp. 7, 10"),
    "learning": (
        "Every operation will connect a calculation to executable code.",
        "Our route begins with algebra and vectors. We then build tensors, neural layers, attention, the two streams, and the complete classifier. For each important operation, we will compare a hand calculation with executable PyTorch. An output that merely has the right shape is not enough.",
        "PDF Appendices C-I; Sec. 16"),
    "exercise1": (
        "Which representation needs two particle addresses?",
        "Here is the first question. Which representation needs two particle addresses, X or R? Consider what one entry describes. Take a short moment to decide. Now follow the highlighted connection: it joins two particles, so the relation representation R needs both particle addresses.",
        "PDF Appendix I, exercise 10"),
    "exercise2": (
        "Does the orange representation change after each particle block?",
        "Second question: does the orange representation change after each particle block? Look at the diagram as the blue states advance. The orange table is reused. Its summaries can depend on the block configuration, but the relation tensor itself is not recurrently rewritten in this implementation.",
        "PDF Appendix I, exercise 10"),
    "summary": (
        "We will explain every operation before assembling the model.",
        "The destination is now clear. A collection of particle observations and pair observations produces one jet prediction. Blue particle states evolve; orange relations are reused. The reported model combines red gates and teal bias. Next, we learn the algebra needed to explain those operations rather than naming them.",
        "PDF pp. 10-12"),
}

CHAPTER_01 = {
    "recap": (
        "Our model transforms particle observations and pair relationships.",
        "Recall the destination. Blue states describe individual particles, and orange states describe their relationships. Before either becomes a neural representation, we need a language for calculations. This chapter introduces that language through a small number of examples that we can calculate completely.",
        "PDF pp. 12, 32-34"),
    "objective": (
        "You will distinguish functions, vectors, and three different products.",
        "By the end, you will evaluate a function, read an indexed sum, and distinguish elementwise, inner, and outer products. You will know the output shape before calculating. That habit prevents a common gating error: accidentally replacing a vector of channel scales with one number.",
        "PDF Appendices C-D"),
    "variable": (
        "A variable names a value that can change.",
        "Start with a simple problem: we want one rule that works for different inputs. The letter x names the input. The letter y names the output. Both are scalars here, meaning single numbers. The constants two and three specify the rule shown on screen.",
        "PDF C.1"),
    "substitute": (
        "Substitution replaces the input symbol with its chosen value.",
        "Set x to four. Replace the input symbol with four without changing the rest of the rule. Multiply two by four to obtain eight, then add three to obtain eleven. The equation now tells us that the output y is eleven. Each step preserves equality.",
        "PDF C.1"),
    "parameters": (
        "A weight scales the input, while a bias shifts output.",
        "Replace the fixed two by a weight w and the fixed three by a bias b. These are scalar parameters in this example. The weight controls scaling; the bias adds a shift. Training may change these parameters. Evaluating the rule with fixed parameters is not itself training.",
        "PDF C.1, C.5"),
    "assignment": (
        "An assignment updates a value rather than asserting equality.",
        "There is a different meaning behind the left-pointing update arrow. It tells us to replace an old value with a new one. An equals sign asserts that two expressions have the same value. Later, residual blocks use updates. Do not read an update as an algebraic contradiction.",
        "PDF C.1"),
    "code_scalar": (
        "The code follows the same multiplication and addition.",
        "Here is executable PyTorch for the scalar calculation. The tensor holds four. Multiplication produces eight, and addition produces eleven. The printed output is generated by running this code during the checks, not typed as an unverified claim. Compare every code line with its matching mathematical step.",
        "PDF C.1; computed code equivalent"),
    "function": (
        "A function maps an input to an output.",
        "Now give a rule a name. The function f doubles its input and adds one. The function g squares its input. These names let us discuss the operations without rewriting them every time. Each function accepts a scalar and returns a scalar in this example.",
        "PDF C.2"),
    "composition": (
        "Composition feeds one function's output into another function.",
        "Apply f first. Its output becomes the input to g. We therefore replace the input of the squaring function by the entire expression two x plus one. The result is the square of that entire expression. The parentheses matter because both terms must be squared together.",
        "PDF C.2"),
    "order": (
        "Changing the order generally changes the resulting function.",
        "Reverse the order. Squaring first gives x squared, and applying f then doubles that square and adds one. This is not the same expression as squaring two x plus one. A neural network is a composition too. Reordering its operations can change its behaviour.",
        "PDF C.2; algebraic consequence"),
    "code_function": (
        "The code preserves the order of the nested functions.",
        "Using four as an illustrative input, the inner function f returns nine, and the outer function g returns eighty one. Reversing the order gives thirty three. These numbers are calculated from the report's functions. They are not trained-model measurements or experimental performance results.",
        "PDF C.1-C.2; derived illustrative outputs"),
    "vector": (
        "A vector is an ordered list of numbers.",
        "One scalar cannot store several observations at once. A vector collects numbers in an order. The report's example contains two, four, six, and eight. We write it vertically as a column vector. Its shape is four: there are four entries, not four separate examples.",
        "PDF D.1"),
    "index": (
        "An index selects an entry rather than multiplying it.",
        "Lowercase c is the channel index. The subscript c selects one entry of x. When c is one, we select two. When c is three, we select six. The subscript does not multiply x by c. It is an address within the ordered list.",
        "PDF C.3, D.1"),
    "sum": (
        "A summation adds the entries selected by its index.",
        "The sigma symbol abbreviates repeated addition. Start the index at one and stop at D, the vector length. For this four-entry vector, the expression expands to two plus four plus six plus eight. Add the terms in sequence: six, twelve, then twenty.",
        "PDF C.3"),
    "mean": (
        "The arithmetic mean divides the sum by the entry count.",
        "An average uses both the sum and the count. The sum is twenty and the count is four, so the arithmetic mean is five. Do not divide by an unrelated dimension such as batch size. The denominator must count the entries included in this particular sum.",
        "PDF C.3"),
    "code_sum": (
        "The reduction removes the axis that it sums.",
        "The code stores the four entries in one tensor. Calling sum returns twenty, and mean returns five. Both outputs are scalars because the single entry axis has been reduced. The original tensor remains a vector. A reduction does not change the stored input in place.",
        "PDF C.3; PyTorch equivalent"),
    "add_scale": (
        "Addition and scalar multiplication act on corresponding entries.",
        "For vectors, addition combines matching positions. Scaling by a scalar multiplies every entry by the same number. The report writes these rules using a channel index. The output retains the vector's shape. Matching the positions is essential: the order defines which entries correspond.",
        "PDF D.1"),
    "products_intro": (
        "The multiplication symbol must specify which operation you mean.",
        "We now meet three operations that are often confused. The report uses a equal to one, two, and b equal to three, four. Both are vectors of length two. We will use the same inputs for all three products, so only the operation changes.",
        "PDF D.3"),
    "hadamard_first": (
        "Elementwise multiplication pairs entries in the same position.",
        "For the elementwise product, connect the first entry of a to the first entry of b. One times three gives three. Then connect the second entries. Two times four gives eight. Keep those two products as separate entries. Do not add them yet.",
        "PDF D.3"),
    "hadamard_result": (
        "The Hadamard product returns a vector of matching products.",
        "The output is the vector three, eight. Its shape is still two. The circled dot denotes this Hadamard product. This is the operation used when a query vector is multiplied by its gate. Each channel keeps its own scaled value, so the channel axis survives.",
        "PDF D.3"),
    "code_hadamard": (
        "PyTorch uses an asterisk for elementwise tensor multiplication.",
        "In PyTorch, a times b, written with an asterisk, performs this elementwise operation for these equal-shaped tensors. Running the code returns three and eight as two entries. A common mistake is to use matrix multiplication here. That would perform a different operation and change the output shape.",
        "PDF D.3; code gds_part_colab.py gate application"),
    "dot_first": (
        "The inner product adds the matching products.",
        "The dot product begins with exactly the same two pairwise multiplications. One times three is three, and two times four is eight. This time, add the results. Three plus eight gives eleven. The final output is one scalar, not a vector with two entries.",
        "PDF D.2-D.3"),
    "dot_notation": (
        "For these real vectors, dot product means inner product.",
        "The report writes the dot product as a transposed vector multiplied by another vector. Transpose turns a column into a row. A one-by-two row times a two-by-one column produces one entry. Equivalently, sum the channelwise products using the index c from one to D.",
        "PDF D.2-D.3"),
    "code_dot": (
        "The code confirms that the result is one scalar.",
        "PyTorch dot returns eleven for these vectors. Summing their elementwise product gives the same answer. The empty shape shown by the check denotes a scalar tensor. Do not confuse that empty shape with an empty vector: this tensor contains one number, eleven.",
        "PDF D.2-D.3; computed code equivalent"),
    "length": (
        "A vector's length is the square root of its self-product.",
        "The Euclidean norm gives the vector's length. Dot the vector with itself, then take a square root. The report also relates a dot product to the lengths of both vectors and the cosine of their angle. Therefore a dot product depends on magnitude as well as alignment.",
        "PDF D.2"),
    "outer_first": (
        "The outer product multiplies every entry against every entry.",
        "The outer product asks a different question. Instead of pairing only matching positions, pair every entry of a with every entry of b. The first row uses the first entry of a: one times three, then one times four. This row becomes three, four.",
        "PDF D.3"),
    "outer_second": (
        "Each row fixes one entry from the first vector.",
        "The second row uses the second entry of a. Two times three gives six, and two times four gives eight. Place those products in the second row. The completed output has two rows and two columns. Its four entries retain all pairwise combinations.",
        "PDF D.3"),
    "outer_result": (
        "The outer product produces a matrix, not a scalar.",
        "The notation is a column vector a multiplied by the row vector b transpose. The result is a two-by-two matrix. Compare it with the inner product: there, the transpose was on a and the output was scalar. The position of that transpose changes the computation.",
        "PDF D.3"),
    "code_outer": (
        "The outer function computes the complete grid of products.",
        "PyTorch outer constructs the same matrix. Its first row is three, four, and its second row is six, eight. Inspect the printed shape: two by two. This is why a successful multiplication alone is not enough. We must verify that it is the intended multiplication.",
        "PDF D.3; computed code equivalent"),
    "compare": (
        "The same inputs can produce three different output shapes.",
        "Place the three results together. Hadamard multiplication keeps two channelwise products. The inner product sums those products into eleven. The outer product stores four combinations in a matrix. For equal-length vectors, the Hadamard result is also the diagonal of this outer-product matrix.",
        "PDF D.3"),
    "gds": (
        "Gates preserve channels before attention reduces them.",
        "Return to the model map. A query vector q and its gate g Q have matching channel shapes. Their elementwise product produces a modified query with the same shape. Later, attention dots that query with a key vector to produce one score. Gating and scoring are different steps.",
        "PDF D.3, Sec. 7.4"),
    "exercise_pause": (
        "Compute all three products before watching the solution.",
        "Now solve the report's first exercise. Use a equal to two, minus one, and b equal to three, four. Predict the shape of each result before calculating. We will pause briefly, then work every multiplication and addition on screen. Pause longer yourself if you want.",
        "PDF Appendix I, exercise 1"),
    "exercise_hadamard": (
        "Matching positions give six and negative four.",
        "First calculate the Hadamard product. Two times three gives six in the first position. Minus one times four gives minus four in the second position. Keep the two numbers separate. The result is the vector six, minus four. Its shape is two.",
        "PDF Appendix I, answer 1"),
    "exercise_inner": (
        "Adding the matching products gives the scalar two.",
        "For the inner product, reuse the two matching products, six and minus four. Add them: six minus four equals two. The result is scalar. The negative second contribution reduces the total. This signed arithmetic matters later when we explain why larger gates need not increase attention.",
        "PDF Appendix I, answer 1; Sec. 8"),
    "exercise_outer": (
        "All four combinations form a two-by-two matrix.",
        "For the outer product, the first row is two times three and two times four, giving six and eight. The second row is minus one times three and minus one times four, giving minus three and minus four. None of these four entries is summed away.",
        "PDF Appendix I, answer 1"),
    "exercise_code": (
        "Running the exercise code reproduces all three answers.",
        "The executable exercise confirms the vector, scalar, and matrix. Compare the minus signs carefully with the hand calculation. If your code produces only one number for the gate operation, you used a reduction too early. If it produces a matrix, you computed all combinations instead.",
        "PDF Appendix I, answer 1; computed code equivalent"),
    "exercise_order": (
        "Explain why reversing the functions changes their composition.",
        "The second question is conceptual. Why does reversing f and g change the result? In one order, the addition happens before squaring and participates in the square. In the other order, the addition happens afterward. The diagram makes that difference visible without appealing to memorised terminology.",
        "PDF C.2"),
    "exercise_shape": (
        "Choose the product that preserves a gate's channel vector.",
        "Third question: which product should implement a per-channel gate? The output must retain one entry per channel, so use the Hadamard product. The inner product removes the channel axis, and the outer product introduces a second axis. The desired meaning determines the correct operation.",
        "PDF D.3"),
    "summary": (
        "Shapes help distinguish operations before any code is written.",
        "A variable names a value, a function transforms it, and composition chains transformations in a specified order. A vector stores ordered entries. Hadamard, inner, and outer products return a vector, scalar, and matrix respectively. Next, we extend this shape reasoning to batches, tensors, and masks.",
        "PDF Appendices C-E"),
}

SUMMARY_00 = [
    "Particles and relations form separate representations.",
    "Relations are reused while particle states evolve.",
    "The reported model combines gates and pair bias.",
]
SUMMARY_01 = [
    "Composition applies functions in a specified order.",
    "A product's output shape identifies its operation.",
    "Gates scale channels without summing them.",
]

CODE = {
    "scalar": "import torch\nx = torch.tensor(4.)\ny = 2*x + 3\nprint(y)",
    "function": "import torch\nx = torch.tensor(4.)\nf = lambda x: 2*x + 1\ng = lambda x: x*x\nprint(g(f(x)), f(g(x)))",
    "sum": "import torch\nx = torch.tensor([2.,4.,6.,8.])\nprint(x.sum(), x.mean())",
    "hadamard": "import torch\na = torch.tensor([1.,2.])\nb = torch.tensor([3.,4.])\nprint(a*b)",
    "dot": "import torch\na = torch.tensor([1.,2.])\nb = torch.tensor([3.,4.])\nprint(torch.dot(a,b))\nprint(torch.dot(a,b).shape)",
    "outer": "import torch\na = torch.tensor([1.,2.])\nb = torch.tensor([3.,4.])\nprint(torch.outer(a,b))",
    "exercise": "import torch\na = torch.tensor([2.,-1.])\nb = torch.tensor([3.,4.])\nprint(a*b)\nprint(torch.dot(a,b))\nprint(torch.outer(a,b))",
}

TEXT_COMMON = [
    "This preview has no narration.",
    "The voice is generated using neural speech synthesis.",
    "Report vs code difference",
    "B counts jets in a batch.",
    "N counts padded particle slots.",
    "D counts particle channels.",
    "Dp counts relation channels.",
    "X represents particle states.",
    "R represents pair relations.",
    "G represents channel gates.",
    "B represents additive pair bias.",
    "q represents a query vector.",
    "k represents a key vector.",
    "g represents a gate vector.",
    "x and y are scalar input and output.",
    "w is a scalar weight.",
    "b is a scalar bias.",
    "f and g map scalars to scalars.",
    "c selects one channel.",
    "D counts vector entries.",
    "a and b are two-entry column vectors.",
    "Transpose exchanges rows and columns.",
    "The norm measures Euclidean length.",
    "Theta represents the angle between nonzero vectors.",
    "These are computed examples, not trained-model measurements.",
    "Multiplication must preserve the intended output shape.",
]
