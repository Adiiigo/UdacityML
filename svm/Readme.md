## Steps to execute Support Vector Machines


1. python svm_author_id.py
    - ##### Ouput1(C=1000,kernel=linear)
        no. of Chris training emails: 7936

        no. of Sara training emails: 7884

        Training Time =  136.697 s

        Test Time =  13.276 s

        Accuracy =  0.994880546075


2. python svm_author_id.py
   - ##### Output2 - After adding lines to decrease the time (C=1000 , kernel=linear)
        no. of Chris training emails: 7936

        no. of Sara training emails: 7884

        Training Time =  0.336 s

        Test Time =  1.288 s

        Accuracy =  0.8606370876


3. python svm_author_id.py
    - ##### Output3 (C=1 , kernel=linear)
        no. of Chris training emails: 7936

        no. of Sara training emails: 7884

        Training Time =  0.181 s

        Test Time =  1.97 s

        Accuracy =  0.884527872582


4. python svm_rbf_author_id.py
   - ##### Output4 (C=1 , kernel=rbf)
        no. of Chris training emails: 7936

        no. of Sara training emails: 7884

        Training Time =  0.165 s

        Test Time =  1.912 s

        Accuracy =  0.616040955631


5. python svm_author_id_class.py
   - ##### Output5 (C=10000 , Kernel=rbf , gamma=auto)
       no. of Chris training emails: 7936

       no. of Sara training emails: 7884

       Training Time =  0.158 s

       Test Time =  1.488 s

       Class of

       10th element =  1

       26th element =  0

       50th element =  1

       Accuracy =  0.892491467577
       

5. python svm_author_id_count.py
   - ##### Output5 (C=10000 , Kernel=rbf , gamma=auto)
        no. of Chris training emails: 7936
        
        no. of Sara training emails: 7884
        
        Training Time =  0.161 s
        
        Test Time =  1.495 s
        
        Chris' mail count =  1018
        
        Accuracy =  0.892491467577


_**-- Greater the value of C , better the accuracy**_
