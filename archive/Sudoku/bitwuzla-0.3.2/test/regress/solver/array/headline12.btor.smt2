(set-logic QF_ABV)
(set-info :status sat)
(declare-const a0 (Array (_ BitVec 16) (_ BitVec 32) ))
(declare-const v0 (_ BitVec 32))
(declare-const v1 (_ BitVec 32))
(declare-const v2 (_ BitVec 32))
(declare-const v3 (_ BitVec 16))
(assert (= #b1 (ite (distinct (select a0 v3) (bvmul (bvmul v1 v2) (bvmul (bvmul v0 v1) (bvmul v0 v2)))) #b1 #b0)))
(check-sat)