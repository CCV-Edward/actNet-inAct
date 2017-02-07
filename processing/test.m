% load('data.mat');

% preds = preds(:,1:200);
% for i = 1:length(preds)
%     preds(i,:) = preds(i,:) - min(preds(i,:))+1;
%     preds(i,:) = clscores.*preds(i,:);
% %     preds(i,:) = preds(i,:)/sum(preds(i,:));
%     preds(i,:) = preds(i,:).*predEXT(indexs(i),1:200);
% %     preds(i,:) = preds(i,:)/max(preds(i,:));
%     preds(i,:) = preds(i,:)/sum(preds(i,:));
% %     preds(i,:) = preds(i,:)/sum(preds(i,:).*preds(i,:));
% end
load('colScoreSmoothed.mat')
sortedScores = sort(binRF);
offset = uint8(length(binRF)*0.15);
minS = mean(sortedScores(1:offset));
% sortedScores = sortedScores;
maxS = mean(sortedScores(end-offset:end));
binRF = (binRF-minS)/(maxS-minS);
