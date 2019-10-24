mkdir -p out

dt=$(date '+%d%m%Y%H%M%S');
echo $dt

iteration_target=2000

for bs in 50 100 200 400 500 1000
do
for lr in 0.0001 0.0003
do

exp=checkers
zdim=2
wae_lambda=100.0
train_size=10000
folc=1
epoch_num=$((iteration_target * bs / train_size))
mmd_or_sinkhorn='mmd'

pz=uniform
rec_lambda=1.0
sinkhorn_iters=10
sinkhorn_epsilon=0.01
ot_lambda=0
nat_resampling=None
# loss_mmd (a monitored but not optimized value) is calculated with this many prior samples:
nat_size=${train_size}
mmd_linear=False


name="wae-mmd_res=${nat_resampling}_exp=${exp}_batch_size=${bs}_lr=${lr}_train_size=${train_size}_rec_lambda=${rec_lambda}_sinkhorn_epsilon=${sinkhorn_epsilon}_ot_lambda=${ot_lambda}_mmd_or_sinkhorn=${mmd_or_sinkhorn}_${dt}"
CUDA_VISIBLE_DEVICES=3 python run.py --pz=${pz} --train_size=${train_size} --mmd_or_sinkhorn=${mmd_or_sinkhorn} \
    --mmd_linear=${mmd_linear} --sinkhorn_sparse=False --enc_noise=deterministic --name="${name}" \
    --nat_size=${nat_size} --nat_resampling=${nat_resampling} --tag="syn,syn_2c,wae,bstuning5" \
    --exp=${exp} --rec_lambda=${rec_lambda} --wae_lambda=${wae_lambda} --epoch_num=${epoch_num} \
    --ot_lambda=${ot_lambda} --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} \
    --frequency_of_latent_change=${folc} \
    --zdim=${zdim} --e_pretrain=False --batch_size=${bs} --lr=${lr} --work_dir=out/sinkhorn_${name} \
        > out/${name}.cout 2> out/${name}.cerr
done
done
