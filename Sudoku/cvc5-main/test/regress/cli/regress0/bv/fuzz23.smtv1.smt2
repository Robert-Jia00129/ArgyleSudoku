(set-option :incremental false)
(set-info :status unsat)
(set-logic QF_BV)
(declare-fun v0 () (_ BitVec 4))
(declare-fun v1 () (_ BitVec 4))
(declare-fun v2 () (_ BitVec 4))
(check-sat-assuming ( (let ((_let_0 (ite (bvslt v2 v2) (_ bv1 1) (_ bv0 1)))) (let ((_let_1 ((_ repeat 1) (ite (= (_ bv1 1) ((_ extract 3 3) v2)) v1 (_ bv1 4))))) (let ((_let_2 (bvadd ((_ zero_extend 3) (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1))) v2))) (let ((_let_3 (bvshl (_ bv1 4) v1))) (let ((_let_4 (bvnot _let_3))) (let ((_let_5 (ite (bvsge ((_ repeat 1) (_ bv2 4)) ((_ sign_extend 3) (ite (bvslt _let_1 _let_1) (_ bv1 1) (_ bv0 1)))) (_ bv1 1) (_ bv0 1)))) (let ((_let_6 ((_ sign_extend 0) _let_2))) (let ((_let_7 (bvlshr v1 _let_6))) (let ((_let_8 (bvor ((_ sign_extend 3) _let_0) v1))) (let ((_let_9 (bvashr v2 ((_ sign_extend 3) (ite (bvuge _let_2 (_ bv11 4)) (_ bv1 1) (_ bv0 1)))))) (let ((_let_10 ((_ sign_extend 3) (bvcomp _let_1 _let_2)))) (let ((_let_11 (ite (bvsle _let_10 (_ bv2 4)) (_ bv1 1) (_ bv0 1)))) (let ((_let_12 ((_ zero_extend 3) (ite (bvslt (bvshl v1 (_ bv1 4)) _let_3) (_ bv1 1) (_ bv0 1))))) (let ((_let_13 (bvnor _let_8 _let_6))) (let ((_let_14 (bvmul _let_2 v2))) (let ((_let_15 ((_ sign_extend 0) (_ bv11 4)))) (let ((_let_16 (ite (bvuge ((_ zero_extend 3) (ite (bvuge _let_2 (_ bv11 4)) (_ bv1 1) (_ bv0 1))) _let_9) (_ bv1 1) (_ bv0 1)))) (let ((_let_17 (ite (bvslt ((_ repeat 1) (_ bv2 4)) ((_ zero_extend 3) (bvneg (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1))))) (_ bv1 1) (_ bv0 1)))) (let ((_let_18 ((_ zero_extend 3) _let_5))) (let ((_let_19 (bvxor v2 _let_10))) (let ((_let_20 (ite (bvsle ((_ zero_extend 3) (bvcomp v2 ((_ repeat 1) _let_3))) _let_8) (_ bv1 1) (_ bv0 1)))) (let ((_let_21 (bvor ((_ sign_extend 3) _let_17) ((_ repeat 1) (_ bv2 4))))) (let ((_let_22 ((_ extract 0 0) _let_11))) (let ((_let_23 ((_ rotate_right 2) ((_ repeat 1) _let_6)))) (let ((_let_24 (bvnand _let_9 _let_6))) (let ((_let_25 (bvnor _let_19 v0))) (let ((_let_26 (distinct _let_19 ((_ repeat 1) (_ bv2 4))))) (let ((_let_27 (bvule _let_1 _let_21))) (let ((_let_28 ((_ zero_extend 3) (ite (= (_ bv1 1) ((_ extract 0 0) (bvsub _let_4 ((_ sign_extend 3) (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1)))))) (ite (bvsge _let_12 _let_2) (_ bv1 1) (_ bv0 1)) _let_17)))) (let ((_let_29 (bvsgt (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1)) (ite (= (_ bv1 1) ((_ extract 0 0) (bvsub _let_4 ((_ sign_extend 3) (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1)))))) (ite (bvsge _let_12 _let_2) (_ bv1 1) (_ bv0 1)) _let_17)))) (let ((_let_30 (bvult _let_14 ((_ repeat 1) (_ bv2 4))))) (let ((_let_31 (bvugt _let_2 ((_ zero_extend 3) (ite (bvslt _let_1 _let_1) (_ bv1 1) (_ bv0 1)))))) (let ((_let_32 ((_ zero_extend 3) (ite (bvuge _let_5 _let_0) (_ bv1 1) (_ bv0 1))))) (let ((_let_33 (distinct _let_8 _let_19))) (let ((_let_34 (bvsgt _let_25 (ite (= (_ bv1 1) ((_ extract 3 3) v2)) v1 (_ bv1 4))))) (let ((_let_35 (bvsgt (bvneg (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1))) _let_0))) (let ((_let_36 ((_ zero_extend 3) (bvneg (bvcomp _let_1 _let_2))))) (let ((_let_37 (bvsgt _let_13 _let_36))) (let ((_let_38 (bvsgt _let_4 ((_ zero_extend 3) _let_11)))) (let ((_let_39 (bvsle _let_23 (_ bv1 4)))) (let ((_let_40 ((_ sign_extend 3) _let_20))) (let ((_let_41 (bvsge _let_13 ((_ zero_extend 3) (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1)))))) (let ((_let_42 (bvsge _let_3 (ite (= (_ bv1 1) ((_ extract 3 3) v2)) v1 (_ bv1 4))))) (let ((_let_43 (bvsgt _let_13 _let_7))) (let ((_let_44 (bvsge (_ bv11 4) _let_24))) (let ((_let_45 (bvslt v0 _let_4))) (let ((_let_46 (= ((_ repeat 1) _let_3) (bvsub _let_4 ((_ sign_extend 3) (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1))))))) (let ((_let_47 (bvult (_ bv11 4) _let_23))) (let ((_let_48 (bvuge _let_2 v2))) (let ((_let_49 ((_ zero_extend 3) (ite (bvsge _let_12 _let_2) (_ bv1 1) (_ bv0 1))))) (let ((_let_50 (bvsle (ite (bvuge _let_2 (_ bv11 4)) (_ bv1 1) (_ bv0 1)) (bvcomp _let_1 _let_2)))) (let ((_let_51 ((_ zero_extend 3) _let_20))) (let ((_let_52 (bvsge ((_ sign_extend 3) (ite (bvsge _let_12 _let_2) (_ bv1 1) (_ bv0 1))) v1))) (let ((_let_53 ((_ zero_extend 3) (ite (distinct ((_ zero_extend 0) v1) _let_18) (_ bv1 1) (_ bv0 1))))) (let ((_let_54 (bvule (bvneg (bvcomp _let_1 _let_2)) (ite (bvuge _let_2 (_ bv11 4)) (_ bv1 1) (_ bv0 1))))) (let ((_let_55 (bvugt (bvshl v1 (_ bv1 4)) _let_10))) (let ((_let_56 (bvult (ite (= (_ bv1 1) ((_ extract 3 3) v2)) v1 (_ bv1 4)) _let_23))) (let ((_let_57 (bvsge _let_28 _let_21))) (let ((_let_58 (bvslt _let_8 (ite (= (_ bv1 1) ((_ extract 3 3) v2)) v1 (_ bv1 4))))) (let ((_let_59 (bvugt _let_53 _let_15))) (let ((_let_60 (bvsgt _let_14 ((_ repeat 1) _let_3)))) (let ((_let_61 (bvslt (bvcomp v2 ((_ repeat 1) _let_3)) _let_0))) (let ((_let_62 (bvule v0 ((_ zero_extend 3) (ite (bvslt _let_1 _let_1) (_ bv1 1) (_ bv0 1)))))) (let ((_let_63 (bvule ((_ zero_extend 3) _let_16) _let_23))) (let ((_let_64 (bvsle _let_53 (ite (= (_ bv1 1) ((_ extract 3 3) v2)) v1 (_ bv1 4))))) (let ((_let_65 (bvuge _let_17 _let_22))) (let ((_let_66 (not _let_37))) (let ((_let_67 (not _let_62))) (let ((_let_68 (not _let_46))) (let ((_let_69 (not (distinct _let_7 _let_49)))) (let ((_let_70 (not (bvuge _let_5 _let_22)))) (let ((_let_71 (not (distinct (bvsub _let_4 ((_ sign_extend 3) (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1)))) (_ bv2 4))))) (let ((_let_72 (not _let_48))) (let ((_let_73 (not (= _let_2 _let_4)))) (let ((_let_74 (not (bvsge _let_1 _let_12)))) (let ((_let_75 (not (bvsle _let_23 (bvsub _let_4 ((_ sign_extend 3) (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1)))))))) (let ((_let_76 (not (bvult _let_49 _let_3)))) (let ((_let_77 (not (bvsgt v0 _let_12)))) (let ((_let_78 (not (bvsge ((_ repeat 1) (_ bv2 4)) _let_13)))) (let ((_let_79 (not _let_26))) (let ((_let_80 (not (bvsle ((_ sign_extend 3) _let_11) _let_6)))) (let ((_let_81 (not (bvugt _let_40 (_ bv11 4))))) (let ((_let_82 (not (bvugt (_ bv1 4) v0)))) (let ((_let_83 (not (distinct _let_18 _let_4)))) (and (or _let_66 _let_31 (bvuge _let_5 _let_22)) (or _let_54 _let_67 (bvsgt (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1)) _let_0)) (or (not (bvslt _let_25 _let_14)) _let_33 (not _let_30)) (or _let_55 _let_34 (not _let_61)) (or _let_68 (not _let_50) (= ((_ zero_extend 0) v1) _let_32)) (or _let_69 _let_70 _let_71) (or (= _let_2 _let_4) _let_72 (not (bvsge _let_1 _let_1))) (or (not _let_27) _let_59 _let_43) (or (bvsge ((_ repeat 1) (_ bv2 4)) _let_13) _let_64 (not _let_52)) (or _let_48 (not _let_38) (not (bvsgt _let_9 ((_ sign_extend 3) (ite (bvslt (bvshl v1 (_ bv1 4)) _let_3) (_ bv1 1) (_ bv0 1)))))) (or _let_66 _let_30 _let_46) (or (= (_ bv11 4) ((_ zero_extend 3) (bvcomp _let_1 _let_2))) (not _let_43) _let_73) (or _let_74 _let_75 _let_75) (or _let_45 (bvugt _let_40 (_ bv11 4)) _let_50) (or (bvsle (bvcomp _let_1 _let_2) _let_11) (= (bvshl v1 (_ bv1 4)) ((_ sign_extend 3) (bvneg (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1))))) _let_57) (or _let_73 (not (distinct _let_25 ((_ sign_extend 3) _let_11))) _let_38) (or _let_76 _let_62 _let_29) (or _let_42 _let_56 _let_74) (or _let_47 _let_56 (= _let_21 ((_ repeat 1) _let_6))) (or _let_77 (not _let_57) (not _let_44)) (or _let_78 _let_79 _let_76) (or _let_65 (not _let_60) (not _let_31)) (or _let_80 (distinct _let_32 v1) _let_72) (or _let_67 _let_68 _let_63) (or (not _let_34) _let_39 _let_27) (or _let_45 (not _let_65) _let_37) (or _let_78 (not (bvuge ((_ sign_extend 3) (ite (distinct ((_ zero_extend 0) v1) _let_18) (_ bv1 1) (_ bv0 1))) _let_4)) _let_81) (or _let_52 (not (bvult (ite (bvslt (bvshl v1 (_ bv1 4)) _let_3) (_ bv1 1) (_ bv0 1)) _let_11)) (not _let_59)) (or _let_44 (= _let_7 ((_ zero_extend 3) (bvcomp v2 ((_ repeat 1) _let_3)))) (not (bvsle _let_9 _let_53))) (or _let_54 (not (bvsgt ((_ zero_extend 3) (ite (bvslt _let_1 _let_1) (_ bv1 1) (_ bv0 1))) (bvshl v1 (_ bv1 4)))) (not _let_56)) (or _let_62 _let_68 _let_74) (or (bvule _let_3 _let_32) (not (bvule _let_51 _let_19)) (distinct ((_ sign_extend 3) (ite (bvslt _let_1 _let_1) (_ bv1 1) (_ bv0 1))) _let_8)) (or _let_58 _let_81 _let_60) (or _let_55 (distinct _let_36 _let_15) _let_75) (or (not _let_58) (not (distinct (_ bv2 4) ((_ repeat 1) (_ bv2 4)))) _let_82) (or _let_52 _let_80 _let_46) (or _let_41 (not (bvslt (bvneg (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1))) (bvneg (ite (bvuge ((_ repeat 1) (_ bv2 4)) ((_ repeat 1) (_ bv2 4))) (_ bv1 1) (_ bv0 1))))) _let_47) (or _let_62 (not _let_47) _let_83) (or _let_42 _let_39 (not _let_35)) (or (not (bvsgt _let_21 (bvshl v1 (_ bv1 4)))) _let_37 _let_71) (or _let_83 _let_26 _let_79) (or _let_55 (bvslt _let_28 _let_7) _let_41) (or (not (bvule _let_2 _let_19)) (bvsgt _let_12 _let_4) (not (bvuge _let_24 ((_ sign_extend 3) (ite (bvslt _let_1 _let_1) (_ bv1 1) (_ bv0 1)))))) (or (not (bvult _let_16 _let_22)) (bvule _let_23 _let_24) _let_56) (or _let_61 _let_29 (not (= (_ bv11 4) _let_8))) (or _let_70 (bvuge (bvmul ((_ zero_extend 3) _let_17) _let_23) _let_9) (not (distinct _let_51 _let_14))) (or _let_30 (not (bvult _let_6 _let_14)) _let_35) (or (not _let_64) (bvult _let_18 _let_6) _let_27) (or _let_29 _let_69 _let_83) (or _let_43 _let_63 (bvsge _let_40 _let_19)) (or _let_80 _let_82 (not (bvult _let_32 _let_24))) (or _let_76 (not _let_33) _let_77)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))) ))