test output
python train_han_batch_nor.py --mode=test --buckets=[dataset folder] --checkpointDir=[checkpoint folder] --testOutputDir=[test output store folder,default is ./]

Will generate two pkl under testOutputDir:test_output.pkl,embedding.pkl
In test_output.pkl: one list including all test result,within each element:
[0] positive sample score
[1] negative sample score
[2] cat attn weight for positive sample
[3] cat attn weight for negative sample
[4] item attn weight for positive sample
[5] item attn weight for negative sample
[6] item history
[7] cat history
[8] positive sample item id
[9] negative sample item id

in embedding.pkl"
[0] item embedding
[1] cat embedding


draw auc:
python auc_draw.py --input_file=[std_out.txt]
output:
train_loss.png
test_loss.png
eval_gauc.png
eval_auc.png
