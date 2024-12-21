function clst_idx = funGMM( z, Nclst )
%% by foreseer wang
% web: http://blog.csdn.net/foreseerwang
% QQ: 50834

[len, k]=size(z);

k_idx=kmeans(z,Nclst);     % 用K-means聚类结果做GMM聚类的初始值
mu=zeros(Nclst,k);
sigma=zeros(k,k,Nclst);
for ii=1:Nclst,
    mu(ii,:)=mean(z(k_idx==ii,:));
    sigma(:,:,ii)=cov(z(k_idx==ii,:));
end;
Pw=ones(Nclst,1)*1.0/Nclst;

px=zeros(len,Nclst);
r=zeros(len,Nclst);
for jj=1:1000, % 简单起见，直接循环，不做结束判断
    for ii=1:Nclst,
        px(:,ii)=mvnpdf(z,mu(ii,:),squeeze(sigma(:,:,ii)));
    end;
    
    % E step
    temp=px.*repmat(Pw',len,1);
    r=temp./repmat(sum(temp,2),1,Nclst);

    % M step
    rk=sum(r);
    pw=rk/len;
    mu=r'*z./repmat(rk',1,k);
    for ii=1:Nclst
        sigma(:,:,ii)=z'*(repmat(r(:,ii),1,k).*z)/rk(ii)-mu(ii,:)'*mu(ii,:);
    end;
end;

% display
[dummy,clst_idx]=max(px,[],2);

end
