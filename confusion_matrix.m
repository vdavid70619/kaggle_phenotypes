function [acc] = confusion_matrix(labels, predict_label)
    tp = sum(predict_label==1&labels==1);
    fp = sum(predict_label==1&labels==0);
    tn = sum(predict_label==0&labels==0);
    fn = sum(predict_label==0&labels==1);

    fprintf('------- Confusion Matrix ------- \n');
    fprintf('       1    0                \n');
    fprintf('   1 %4d %4d                 \n', tp, fp);
    fprintf('   0 %4d %4d                 \n', fn, tn);

    precison = tp/(tp+fp);
    recall = tp/(tp+fn);
    fprintf(' [Recall]: %02.04f%%            \n', 100*recall);
    fprintf(' [Precision]: %02.04f%%         \n', 100*precison);
    fprintf(' [F-measurement]: %02.04f%%     \n', 100*2*precison*recall/(precison+recall)); 
    fprintf(' [G-measurement]: %02.04f%%     \n', 100*sqrt(precison*recall));     
    fprintf(' [Accuracy]: %02.04f%%     \n', 100*(tp+tn)/(tp+tn+fp+fn));    
    fprintf('-------------------------------- \n');   
    acc = 100*(tp+tn)/(tp+tn+fp+fn);
end