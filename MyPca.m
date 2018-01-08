function [Eigenvector,ScorePCA,EigenValue] = MyPca(InputMatrix)

    [nrows,ncols] = size(InputMatrix);
    
    MeanMat = zeros(1,ncols);
    
    for x = 1:ncols
        
        MeanMat(x) = mean(InputMatrix(:,x))
        
        for y = 1:nrows
            InputMatrix(y,x) = InputMatrix(y,x)-MeanMat(x);
        end
    end
    
    B = InputMatrix'*InputMatrix;
    [E,V] = eig(B);
    
    ScorePCA = (E*InputMatrix')';
    
    Eigenvector = E;
    EigenValue = V;


end

