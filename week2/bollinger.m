clear;


function ma = movingAvg(x, n)
    ma = movmean(x, [n-1 0]);
end
function s = movingStd(x, n)
    s = movstd(x, [n-1 0]);
end
function y = lag(x,k)
    y = [nan(k,size(x,2)); x(1:end-k,:)];
end
load('inputData_ETF', 'tday', 'syms', 'cl');
idxA=find(strcmp('GLD', syms));
idxB=find(strcmp('USO', syms));

x=cl(:,idxA);
y=cl(:,idxB);

lookback=20; 
h=NaN(size(x, 1), 1);
for t=lookback:size(h, 1)
    regression_result=ols(y(t-lookback+1:t), [x(t-lookback+1:t) ones(lookback, 1)]);
    h(t)=regression_result(1);
end

y2=[x y];

yport=sum([-h ones(size(h))].*y2, 2); 
h(1:lookback)=[]; 
yport(1:lookback)=[];
y2(1:lookback, :)=[];

entryZscore=1;
exitZscore=0;

ma=movingAvg(yport, lookback);
std=movingStd(yport, lookback);
zScore=(yport-ma)./std;

longsEntry=zScore < -entryZscore; 
longsExit=zScore > -exitZscore;

shortsEntry=zScore > entryZscore;
shortsExit=zScore < exitZscore;

numUnitsLong=NaN(length(yport), 1);
numUnitsShort=NaN(length(yport), 1);

numUnitsLong(1)=0;
numUnitsLong(longsEntry)=1;
numUnitsLong(longsExit)=0;
numUnitsLong=fillMissingData(numUnitsLong); 

numUnitsShort(1)=0;
numUnitsShort(shortsEntry)=-1;
numUnitsShort(shortsExit)=0;
numUnitsShort=fillMissingData(numUnitsShort);

numUnits=numUnitsLong+numUnitsShort;
positions=repmat(numUnits, [1 size(y2, 2)]).*[-h ones(size(h))].*y2; 
pnl=sum(lag(positions, 1).*(y2-lag(y2, 1))./lag(y2, 1), 2); 
ret=pnl./sum(abs(lag(positions, 1)), 2); 
ret(isnan(ret))=0;

figure;
plot(cumprod(1+ret)-1);