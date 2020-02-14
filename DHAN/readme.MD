DHAN utilisation manual
======================

Data preparation
-----------------
Train
-----------------

Test
-----------------
python train_han_batch_nor.py --mode=test --buckets=[dataset folder] --checkpointDir=[checkpoint folder] --testOutputDir=[test output store folder,default is ./]<br> 

This will generate two pkl under testOutputDir:test_output.pkl and embedding.pkl <br>  
In test_output.pkl: one list including all test result,within each element:<br>  
[0] positive sample score<br>  
[1] negative sample score<br>  
[2] cat attn weight for positive sample<br>  
[3] cat attn weight for negative sample<br>  
[4] item attn weight for positive sample<br>  
[5] item attn weight for negative sample<br>  
[6] item history<br>  
[7] cat history<br>  
[8] positive sample item id<br>  
[9] negative sample item id<br>  

in embedding.pkl
[0] item embedding<br>  
[1] cat embedding<br>  


Auc draw
---------------
python auc_draw.py --input_file=[std_out.txt] --input_file_din=[std_out.txt]<br>  
output:<br>  
train_loss.png<br>  
test_loss.png<br>  
eval_gauc.png<br>  
eval_auc.png<br>  
