Step 1
给xlsx和病人目录改名，去掉空格并且大写首字母
python dataset/ChengdaOnlyCSPca/rename.py

Step 2 
删除非csPCa文件
python dataset/ChengdaOnlyCSPca/delete.py 


Step 3   !!!!!!!!!!!!! 需要手动
清理病人目录下面的繁杂数据，例如多个DWI序列
3.1 先重命名ADC和T2W目录

Step 4
肉眼找出 b=800 的DWI ， 然后组织成nnUNet数据格式
4._organize_nnunet_style.py


Step 5
执行重采样和配准流程
python  dataset/ChengdaOnlyCSPca/5_batch_deal_mp_for_chengda.py 

记得改为imagesTr和imagesTs,labelsTr和labelsTs