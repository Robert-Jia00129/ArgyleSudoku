(set-logic QF_ABV)
(declare-fun _substvar_74_ () (_ BitVec 4))
(declare-fun _substvar_76_ () (_ BitVec 1))
(declare-fun _substvar_141_ () Bool)
(declare-fun _substvar_150_ () Bool)
(declare-fun _substvar_155_ () Bool)
(declare-fun _substvar_169_ () (_ BitVec 1))
(push 1)
(assert false)
(set-info :status unsat)
(check-sat)
(pop 1)
(define-fun |Scoreboard_h#805| () Bool (and true true true true true true true true true true true true true true true true true true true true true true true true true true (= _substvar_141_ (= _substvar_169_ #b1)) true true true (and true true true true (= (= (bvor _substvar_76_ _substvar_169_) #b1) _substvar_150_) true true true true true true true true)))
(assert (and true true true true true true true true true true true true true true true true true true true true true true true true true true (= _substvar_141_ (= _substvar_169_ #b1)) true true true (and true true true true (= (= (bvor _substvar_76_ _substvar_169_) #b1) _substvar_150_) true true true true true true true true)))
(assert _substvar_141_)
(push 1)
(assert false)
(set-info :status unsat)
(check-sat)
(pop 1)
(assert (and true true true (and _substvar_155_ true true true (= (ite _substvar_150_ _substvar_74_ (_ bv0 4)) (_ bv0 4)) true)))
(set-info :status sat)
(check-sat)
(exit)