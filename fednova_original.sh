python experiments.py --model=simple-cnn \
	--dataset=cifar10 \
	--alg=fednova \
	--lr=0.01 \
	--batch-size=64 \
	--epochs=10 \
	--n_parties=100 \
	--comm_round=500 \
	--partition=noniid-labeldir \
	--beta=0.5 \
	--device='cuda:1' \
	--datadir='./data/' \
	--logdir='./logs/fednova_original' \
	--noise=0 \
	--init_seed=0 \
	--mu=0 \
	--sample=0.1