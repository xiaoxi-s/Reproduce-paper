# Reproduce-paper

This project is aimmed at reproducing the results in [Learning with Bad Training Data via Iterative Trimmed Loss Minimization](https://arxiv.org/abs/1810.11874).

## Libraries

- Pytorch
- Numpy
- scikit-learn

## Progress

 - Linear Regression: Tests on sample sizes & Test on alphas has been finished. 
   - Results of test on sample sizes fit the original data pretty well (Figure provided).
   - Results of test on alphas are not so correct (Figure provided). 
   - Test on "large noise" has not been tested as the definition of "large" noise need further investigation. 

 - WRN: There are some issues with training. 
   - With Dataloader as input, the accuracy is around 50%.
   - With torch.Tensor as input (numpy data converte to torch.Tensor), the accuracy is only 16%.

Therefore, I did not produce any of the results from WRN. 

Note: 

- The sample sizes for Linear Regression do not exceed 10000 because of the possibility the process being killed on my computer. 
- WRN is trained with 200 epoches.
- WRN source is from [link](https://github.com/meliketoy/wide-resnet.pytorch)

Detailed test code can be found in train\_\*.py
