
for sinkhorn_iters in 10 20 50
do
for sinkhorn_epsilon in 0.1 0.01 0.001
do
for ot_lambda in 100.0 1000.0 10000.0
do
name="nobn_logdomain_sinkhorn_iters=${sinkhorn_iters}_sinkhorn_epsilon=${sinkhorn_epsilon}_ot_lambda=${ot_lambda}"
python run.py --train_size 1000 --wae_lambda=0.0 --epoch_num=200 --ot_lambda=${ot_lambda} --enc_noise=deterministic --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} --zdim=10 --e_pretrain=False --work_dir=sinkhorn_${name}  > sinkhorn_${name}.cout 2> sinkhorn_${name}.cerr
done
done
done