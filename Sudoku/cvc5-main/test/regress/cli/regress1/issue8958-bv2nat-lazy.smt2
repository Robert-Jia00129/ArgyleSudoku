(set-logic ALL)
(set-info :status sat)
(declare-const a (_ BitVec 1))
(assert (= (_ bv0 32) ((_ int2bv 32) (bv2nat ((_ zero_extend 7) a)))))
(check-sat)