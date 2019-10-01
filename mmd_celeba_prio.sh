# Global-MMD. This code was run using the following patch applied on 2e3889d:

#-            OT, P_temp, P, f, g, C = sinkhorn.SparseSinkhornLoss(x_latents_with_current_batch, self.nat_targets, sparse_indices=self.nat_sparse_indices, epsilon=decayed_epsilon, niter=opts['sinkhorn_iters'])
#+            assert False, "now it's MMD, sparse version unimplemented"
#+            # OT, P_temp, P, f, g, C = sinkhorn.SparseSinkhornLoss(x_latents_with_current_batch, self.nat_targets, sparse_indices=self.nat_sparse_indices, epsilon=decayed_epsilon, niter=opts['sinkhorn_iters'])
#         else:
#-            OT, P_temp, P, f, g, C = sinkhorn.SinkhornLoss(x_latents_with_current_batch, self.nat_targets, epsilon=decayed_epsilon, niter=opts['sinkhorn_iters'])
#+            print("NOT SINKHORN, MMD")
#+            OT = self.mmd_penalty(x_latents_with_current_batch, self.nat_targets)
#+            # OT, P_temp, P, f, g, C = sinkhorn.SinkhornLoss(x_latents_with_current_batch, self.nat_targets, epsilon=decayed_epsilon, niter=opts['sinkhorn_iters'])
#+            P = tf.zeros(1)
#+            C = tf.zeros(1)



#CUDA_VISIBLE_DEVICES=0 python run.py --exp=celebA --matching_penalty_scope=None --feed_by_score_from_epoch=-1 --train_size=100000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=1.0 --ot_lambda=0.0 --nat_size=5000 --lr=0.001 --tag="junk,no_big_scope,no_prio_train" --name "celeba_imp_bl" > cout_bl 2> cerr_bl &
#sleep 10

#CUDA_VISIBLE_DEVICES=1 python run.py --exp=celebA --matching_penalty_scope=None --feed_by_score_from_epoch=2 --train_size=100000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=1.0 --ot_lambda=0.0 --nat_size=5000 --lr=0.001 --tag="junk,no_big_scope,prio_train" --name "celeba_imp_bl" > cout_nbs_pt 2> cerr_nbs_pt &
#sleep 10

#CUDA_VISIBLE_DEVICES=2 python run.py --exp=celebA --matching_penalty_scope=nat --feed_by_score_from_epoch=2 --train_size=100000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=1.0 --ot_lambda=0.0 --nat_size=5000 --lr=0.001 --tag="junk,big_scope,prio_train" --name "celeba_imp_bl" > cout_bs_pt 2> cerr_bs_pt &
#sleep 10

#CUDA_VISIBLE_DEVICES=3 python run.py --exp=celebA --matching_penalty_scope=nat --feed_by_score_from_epoch=-1 --train_size=100000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=1.0 --ot_lambda=0.0 --nat_size=5000 --lr=0.001 --tag="junk,big_scope,no_prio_train" --name "celeba_imp_bl" > cout_bs_npt 2> cerr_bs_npt &
#sleep 10

CUDA_VISIBLE_DEVICES=4 python run.py --exp=celebA --matching_penalty_scope=None --feed_by_score_from_epoch=-1 --train_size=100000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=100.0 --ot_lambda=0.0 --nat_size=5000 --lr=0.001 --tag="junk,no_big_scope,no_prio_train,l100" --name "celeba_imp_bl" > cout_bl_l100 2> cerr_bl_l100 &
sleep 10

CUDA_VISIBLE_DEVICES=5 python run.py --exp=celebA --matching_penalty_scope=None --feed_by_score_from_epoch=2 --train_size=100000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=100.0 --ot_lambda=0.0 --nat_size=5000 --lr=0.001 --tag="junk,no_big_scope,prio_train,l100" --name "celeba_imp_bl" > cout_nbs_pt_l100 2> cerr_nbs_pt_l100 &
sleep 10
