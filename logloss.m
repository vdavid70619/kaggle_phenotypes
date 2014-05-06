function LogLoss = logloss(y, y_pred)
    epss=0.001; %arbitrary value, may be model tuning parameter  
    y_pred=min(max(y_pred,epss),1-epss);  
    LogLoss=-mean(y.*log(y_pred)+(1-y).*log(1-y_pred));  
end