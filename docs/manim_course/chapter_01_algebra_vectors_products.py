"""Review chapter 1: complete calculations from PDF Appendices C, D and I."""
from studio import *
from content import CHAPTER_01, SUMMARY_01


class Chapter01(TeachingScene):
    chapter = CHAPTER_01
    chapter_number = 1

    def vector_pair(self, a=(1,2), b=(3,4)):
        av = Matrix([[str(v)] for v in a], element_to_mobject_config={"font_size":44}).move_to(LEFT*3.4)
        bv = Matrix([[str(v)] for v in b], element_to_mobject_config={"font_size":44}).move_to(RIGHT*.1)
        al = self.formula("a=",LEFT*4.8)
        bl = self.formula("b=",LEFT*1.3)
        return av,bv,VGroup(al,bl,av,bv)

    def construct(self):
        xstream=self.formula("X",LEFT*2,COLORS["X"],72)
        rstream=self.formula("R",RIGHT*2,COLORS["R"],72)
        self.teach("recap",[lambda: Write(xstream),lambda: Write(rstream)])
        shapes=self.formula(r"(2,)\qquad ()\qquad (2,2)",DOWN*1.5,size=40)
        self.teach("objective",[lambda: Write(shapes),lambda: Circumscribe(shapes)])

        self.wipe()
        equation=self.formula(r"y=2x+3",UP*.5,size=72)
        definition=self.definition("x and y are scalar input and output.")
        self.shape(r"x:()\quad y:()",COLORS["X"])
        self.teach("variable",[lambda: Write(definition),lambda: Write(equation),
                               lambda: Indicate(equation)])
        e1=self.formula(r"y=2(4)+3",UP*.5,size=72)
        e2=self.formula(r"y=8+3",UP*.5,size=72)
        e3=self.formula(r"y=11",UP*.5,size=72)
        self.teach("substitute",[lambda: TransformMatchingTex(equation,e1),
                                 lambda: TransformMatchingTex(e1,e2),
                                 lambda: TransformMatchingTex(e2,e3),
                                 lambda: Circumscribe(e3)],True)
        general=self.formula(r"y=wx+b",UP*.5,size=72)
        weight=self.definition("w is a scalar weight.")
        bias=self.definition("b is a scalar bias.")
        self.teach("parameters",[lambda: TransformMatchingTex(e3,general),
                                 lambda: ReplacementTransform(definition,weight),
                                 lambda: ReplacementTransform(weight,bias),
                                 lambda: Circumscribe(general)])
        assignment=self.formula(r"x\leftarrow x+u",UP*.5,size=64)
        update_def=self.definition("u is an update with the same shape.")
        self.teach("assignment",[lambda: AnimationGroup(ReplacementTransform(general,assignment),
                                                        ReplacementTransform(bias,update_def)),
                                 lambda: Indicate(assignment)])
        self.code_example("code_scalar","scalar")

        self.wipe()
        f=self.formula(r"f(x)=2x+1",LEFT*3+UP*.5)
        g=self.formula(r"g(x)=x^2",RIGHT*3+UP*.5)
        defs=self.definition("f and g map scalars to scalars.")
        self.teach("function",[lambda: Write(defs),lambda: Write(f),lambda: Write(g)])
        composite=self.formula(r"g(f(x))=(2x+1)^2",DOWN*1.0,size=54)
        arrow=Arrow(f.get_right(),g.get_left(),color=COLORS["X"])
        self.teach("composition",[lambda: Create(arrow),
                                  lambda: TransformFromCopy(f,composite),
                                  lambda: Circumscribe(composite)],True)
        reverse=self.formula(r"f(g(x))=2x^2+1",DOWN*1.0,size=54)
        reversearrow=Arrow(g.get_left(),f.get_right(),color=COLORS["R"])
        self.teach("order",[lambda: ReplacementTransform(arrow,reversearrow),
                            lambda: TransformMatchingTex(composite,reverse),
                            lambda: Indicate(reverse)],True)
        self.code_example("code_function","function")

        self.wipe()
        vector=Matrix([[2],[4],[6],[8]],element_to_mobject_config={"font_size":42}).move_to(LEFT*3.4)
        label=self.formula("x=",LEFT*4.8,COLORS["X"])
        self.shape(r"x:(4,)",COLORS["X"])
        self.teach("vector",[lambda: Write(label),lambda: Write(vector),
                             lambda: Circumscribe(vector)])
        cdef=self.definition("c selects one channel.")
        index1=self.formula(r"x_1=2",RIGHT*2+UP*.6)
        index3=self.formula(r"x_3=6",RIGHT*2+UP*.6)
        self.teach("index",[lambda: Write(cdef),
                            lambda: AnimationGroup(Indicate(vector.get_entries()[0]),Write(index1)),
                            lambda: AnimationGroup(Indicate(vector.get_entries()[2]),
                                                    TransformMatchingTex(index1,index3)),
                            lambda: Circumscribe(cdef)])
        ddef=self.definition("D counts vector entries.")
        sigma=self.formula(r"\sum_{c=1}^{D}x_c",RIGHT*2+UP*.5,size=52)
        expansion=self.formula(r"2+4+6+8",RIGHT*2+UP*.5,size=54)
        partial=self.formula(r"6+6+8=12+8=20",RIGHT*2+UP*.5,size=44)
        self.teach("sum",[lambda: AnimationGroup(ReplacementTransform(cdef,ddef),
                                                ReplacementTransform(index3,sigma)),
                          lambda: TransformMatchingTex(sigma,expansion),
                          lambda: LaggedStart(*[Indicate(v) for v in vector.get_entries()],lag_ratio=.2),
                          lambda: TransformMatchingTex(expansion,partial)],True)
        average=self.formula(r"\frac{2+4+6+8}{4}=\frac{20}{4}=5",RIGHT*2+UP*.5,size=46)
        self.teach("mean",[lambda: ReplacementTransform(partial,average),
                           lambda: Circumscribe(average),lambda: Indicate(ddef)],True)
        self.code_example("code_sum","sum")
        self.wipe()
        addrule=self.formula(r"(x+y)_c=x_c+y_c",UP*.7,size=56)
        scalerule=self.formula(r"(\alpha x)_c=\alpha x_c",DOWN*.7,size=56)
        alphadef=self.definition("Alpha is a scalar scaling factor.")
        self.teach("add_scale",[lambda: Write(addrule),lambda: AnimationGroup(Write(alphadef),Write(scalerule)),
                                lambda: Indicate(scalerule)])

        self.wipe()
        av,bv,pair=self.vector_pair()
        abdef=self.definition("a and b are two-entry column vectors.")
        self.shape(r"a:(2,)\quad b:(2,)")
        self.teach("products_intro",[lambda: Write(abdef),lambda: Write(pair),
                                     lambda: Circumscribe(pair)])
        first=self.formula(r"1\cdot3=3",RIGHT*3.6+UP*.6,size=50)
        second=self.formula(r"2\cdot4=8",RIGHT*3.6+DOWN*.7,size=50)
        self.teach("hadamard_first",[
            lambda: AnimationGroup(Indicate(av.get_entries()[0]),Indicate(bv.get_entries()[0])),
            lambda: TransformFromCopy(VGroup(av.get_entries()[0],bv.get_entries()[0]),first),
            lambda: TransformFromCopy(VGroup(av.get_entries()[1],bv.get_entries()[1]),second),
            lambda: Indicate(VGroup(first,second))],True)
        result=Matrix([[3],[8]],element_to_mobject_config={"font_size":44}).move_to(RIGHT*3.5)
        symbol=self.formula(r"a\odot b=",RIGHT*1.8+UP*1.8,size=44)
        self.teach("hadamard_result",[
            lambda: AnimationGroup(ReplacementTransform(VGroup(first,second),result),Write(symbol)),
            lambda: Circumscribe(result),
            lambda: result.animate.set_color(COLORS["G"]),
            lambda: Indicate(result)],True)
        self.code_example("code_hadamard","hadamard")

        self.wipe()
        multiply=self.formula(r"(1\cdot3)+(2\cdot4)",UP*.5,size=64)
        products=self.formula(r"3+8",UP*.5,size=64)
        eleven=self.formula("11",UP*.5,size=80)
        self.teach("dot_first",[lambda: Write(multiply),
                                lambda: TransformMatchingTex(multiply,products),
                                lambda: TransformMatchingTex(products,eleven),
                                lambda: Circumscribe(eleven)],True)
        transpose=self.definition("Transpose exchanges rows and columns.")
        rowcol=self.formula(r"\begin{bmatrix}1&2\end{bmatrix}\begin{bmatrix}3\\4\end{bmatrix}=11",UP*.5,size=52)
        notation=self.formula(r"a^\top b=\sum_{c=1}^{D}a_cb_c",UP*.5,size=56)
        self.shape(r"(1,2)\ (2,1)\longrightarrow(1,1)")
        self.teach("dot_notation",[lambda: Write(transpose),
                                   lambda: ReplacementTransform(eleven,rowcol),
                                   lambda: Circumscribe(rowcol),
                                   lambda: ReplacementTransform(rowcol,notation)],True)
        self.code_example("code_dot","dot")
        self.wipe()
        normdef=self.definition("The norm measures Euclidean length.")
        norm=self.formula(r"\|x\|_2=\sqrt{x^\top x}",UP*.7,size=56)
        angle=self.formula(r"\cos\theta=\frac{x^\top y}{\|x\|_2\|y\|_2}",UP*.7,size=56)
        theta=self.definition("Theta represents the angle between nonzero vectors.")
        self.teach("length",[lambda: Write(normdef),lambda: Write(norm),
                             lambda: AnimationGroup(ReplacementTransform(norm,angle),
                                                     ReplacementTransform(normdef,theta)),
                             lambda: Circumscribe(angle)],True)

        self.wipe()
        av,bv,pair=self.vector_pair()
        grid=Matrix([[3,4],[6,8]],element_to_mobject_config={"font_size":44}).move_to(RIGHT*3.8)
        self.shape(r"(2,1)\ (1,2)\longrightarrow(2,2)")
        first_row=self.formula(r"(1\cdot3,\ 1\cdot4)=(3,4)",DOWN*1.8,size=42)
        self.teach("outer_first",[lambda: Write(pair),
                                  lambda: Create(grid.get_brackets()),
                                  lambda: Write(first_row),
                                  lambda: TransformFromCopy(VGroup(av.get_entries()[0],*bv.get_entries()),grid.get_rows()[0])],True)
        second_row=self.formula(r"(2\cdot3,\ 2\cdot4)=(6,8)",DOWN*1.8,size=42)
        self.teach("outer_second",[lambda: ReplacementTransform(first_row,second_row),
                                   lambda: TransformFromCopy(VGroup(av.get_entries()[1],*bv.get_entries()),grid.get_rows()[1]),
                                   lambda: Circumscribe(grid),
                                   lambda: Indicate(grid.get_columns()[1])],True)
        outer=self.formula(r"ab^\top=\begin{bmatrix}3&4\\6&8\end{bmatrix}",DOWN*1.8,size=44)
        self.teach("outer_result",[lambda: ReplacementTransform(second_row,outer),
                                   lambda: Circumscribe(outer),lambda: Indicate(grid)])
        self.code_example("code_outer","outer")
        self.wipe()
        comparison=self.formula(
            r"a\odot b=\begin{bmatrix}3\\8\end{bmatrix}\qquad "
            r"a^\top b=11\qquad ab^\top=\begin{bmatrix}3&4\\6&8\end{bmatrix}",size=44)
        self.teach("compare",[lambda: Write(comparison),lambda: Circumscribe(comparison)])
        self.wipe()
        qdef=self.definition("q represents a query vector.",color=COLORS["Q"])
        gdef=self.definition("g represents a gate vector.",color=COLORS["G"])
        kdef=self.definition("k represents a key vector.",color=COLORS["K"])
        q=self.formula("q",LEFT*3,COLORS["Q"],64)
        gate=self.formula(r"g_Q",LEFT,COLORS["G"],64)
        gated=self.formula(r"{{q'}}={{q}}\odot{{g_Q}}",RIGHT*2,TEXT,52)
        gated.set_color_by_tex("q",COLORS["Q"])
        gated.set_color_by_tex("g_Q",COLORS["G"])
        score=self.formula(r"{{s}}=({{q'}})^\top{{k}}",RIGHT*2,TEXT,52)
        score.set_color_by_tex("q",COLORS["Q"])
        score.set_color_by_tex("k",COLORS["K"])
        self.shape(r"q,g_Q,q',k:(d,)\quad s:()")
        ddef=self.prose("d counts channels within one attention head.",UP*2.35)
        self.teach("gds",[lambda: AnimationGroup(Write(q),Write(qdef),Write(ddef)),
                          lambda: AnimationGroup(Write(gate),ReplacementTransform(qdef,gdef)),
                          lambda: Write(gated),
                          lambda: AnimationGroup(ReplacementTransform(gated,score),
                                                  ReplacementTransform(gdef,kdef))],True)

        self.wipe()
        av,bv,pair=self.vector_pair((2,-1),(3,4))
        self.shape(r"a:(2,)\quad b:(2,)")
        self.teach("exercise_pause",[lambda: Write(pair),lambda: Indicate(pair)])
        h1=self.formula(r"2\cdot3=6",RIGHT*3.6+UP*.7,size=46)
        h2=self.formula(r"(-1)\cdot4=-4",RIGHT*3.6+DOWN*.7,size=42)
        hresult=self.formula(r"a\odot b=(6,-4)",DOWN*1.9,size=48)
        self.teach("exercise_hadamard",[lambda: Write(h1),lambda: Write(h2),
                                        lambda: Write(hresult),lambda: Circumscribe(hresult)],True)
        inner=self.formula(r"a^\top b=6+(-4)=2",DOWN*1.9,size=48)
        self.teach("exercise_inner",[lambda: ReplacementTransform(hresult,inner),
                                     lambda: Circumscribe(inner),lambda: Indicate(h2)],True)
        outersteps=self.formula(r"ab^\top=\begin{bmatrix}2\cdot3&2\cdot4\\(-1)\cdot3&(-1)\cdot4\end{bmatrix}",DOWN*1.7,size=42)
        outeranswer=self.formula(r"ab^\top=\begin{bmatrix}6&8\\-3&-4\end{bmatrix}",DOWN*1.7,size=46)
        self.teach("exercise_outer",[lambda: ReplacementTransform(inner,outersteps),
                                     lambda: TransformMatchingTex(outersteps,outeranswer),
                                     lambda: Circumscribe(outeranswer)],True)
        self.code_example("exercise_code","exercise")
        self.wipe()
        order1=self.formula(r"g(f(x))=(2x+1)^2",UP*.8,size=54)
        order2=self.formula(r"f(g(x))=2x^2+1",DOWN*.8,size=54)
        self.teach("exercise_order",[lambda: Write(order1),lambda: Write(order2),
                                     lambda: Indicate(order1),lambda: Indicate(order2)],True)
        self.wipe()
        right=self.formula(r"(d,)\odot(d,)\longrightarrow(d,)",UP*.6,COLORS["G"],52)
        wrong=self.formula(r"(d,)\cdot(d,)\longrightarrow()",DOWN*.8,size=52)
        self.teach("exercise_shape",[lambda: Write(right),lambda: Circumscribe(right),
                                     lambda: Write(wrong),lambda: Indicate(right)],True)
        self.wipe()
        summary=VGroup(*[self.prose(line,UP*(1-i)) for i,line in enumerate(SUMMARY_01)])
        self.teach("summary",[lambda: Write(summary[0]),lambda: Write(summary[1]),
                              lambda: AnimationGroup(FadeOut(self.caption),Write(summary[2]))])
        self.finish()
