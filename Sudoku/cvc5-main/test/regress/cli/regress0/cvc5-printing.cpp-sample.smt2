; SCRUBBER: sed -e 's/((x.*//g'
; EXPECT: unsat
; EXPECT: sat
(set-option :bv-print-consts-as-indexed-symbols true)
(set-logic QF_AUFBV)
(set-option :produce-models true)
(set-option :incremental true)
(set-option :produce-unsat-assumptions true)
(declare-sort S 0)
(declare-fun x () (_ BitVec 32))
(declare-fun y () (_ BitVec 32))
(declare-fun arr () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun f ((_ BitVec 32) ) S)
(declare-fun s () S)
(declare-fun ind1 () Bool)
(push 1)
(assert ind1)
(assert (= ind1 (= (f x) s)))
(assert (= (f x) s))
(assert (not (=> (= x y) (= (select arr x) (select arr y)))))
(check-sat-assuming (ind1 ))
(pop 1)
(check-sat)
(get-value (x))