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



CUDA_VISIBLE_DEVICES=0 python run.py --exp=celebA --train_size=2000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=0.0 --ot_lambda=100 --nat_size=2000 --lr=0.001 --tag="junk,celeba,mini_celeba,mmd_global" --name "celeba_5000_sphere_mmd_global" > cout 2> cerr &
sleep 10

CUDA_VISIBLE_DEVICES=1 python run.py --exp=celebA --nat_resampling=batch --train_size=2000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=0.0 --ot_lambda=100 --nat_size=2000 --lr=0.001 --tag="junk,celeba,mini_celeba,mmd_global,resampling" --name "celeba_5000_sphere_mmd_global_resampling" > cout.resampling 2> cerr.resampling &
sleep 10


CUDA_VISIBLE_DEVICES=2 python run.py --exp=celebA --nat_resampling=batch --train_size=2000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=0.0 --ot_lambda=100 --nat_size=100 --lr=0.001 --tag="junk,celeba,mini_celeba,mmd_local,resampling" --name "celeba_5000_sphere_mmd_local" > cout.resampling_local 2> cerr.resampling_local &
sleep 10


CUDA_VISIBLE_DEVICES=3 python run.py --exp=celebA --frequency_of_latent_change=1 --train_size=2000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=0.0 --ot_lambda=100 --nat_size=2000 --lr=0.001 --tag="junk,celeba,mini_celeba,mmd_global,uncached" --name "celeba_5000_sphere_mmd_global_uncached" > cout.uncached 2> cerr.uncached &
sleep 10


CUDA_VISIBLE_DEVICES=4 python run.py --exp=celebA --nat_resampling=batch --frequency_of_latent_change=1 --train_size=2000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=0.0 --ot_lambda=100 --nat_size=2000 --lr=0.001 --tag="junk,celeba,mini_celeba,mmd_global,resampling,uncached" --name "celeba_5000_sphere_mmd_global_resampling_uncached" > cout.resampling.uncached 2> cerr.resampling.uncached &
sleep 10


CUDA_VISIBLE_DEVICES=5 python run.py --exp=celebA --z_test=mmd --nat_resampling=batch --train_size=2000 --epoch_num=100 --pz=sphere --enc_noise=deterministic --wae_lambda=100 --ot_lambda=0.0 --nat_size=100 --lr=0.001 --tag="junk,celeba,mini_celeba,mmd_local,baseline,resampling" --name "celeba_5000_sphere_mmd_local_really" > cout.real_mmd 2> cerr.real_mmd &
