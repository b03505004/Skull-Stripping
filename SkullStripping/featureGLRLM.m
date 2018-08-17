function [SRE, LRE, GLN, RLN, RP, LGRE, HGRE, SRLGE, SRHGE, LRLGE, LRHGE] = featureGLRLM(imgNoisy,maxPossible)

imgNoisy = 255.*imgNoisy /maxPossible;

%angle: 0, 45, 90, 135
offsets = [1;2;3;4];

%compute run-length matrixs
[GLRLMS,SI] = grayrlmatrix(imgNoisy,'Offset',offsets, 'NumLevels', 8, 'G', [min(imgNoisy(:)) max(imgNoisy(:))]);

%compute 11 features
features = grayrlprops(GLRLMS);

SRE = features(:,1);
LRE = features(:,2);
GLN = features(:,3);
RLN = features(:,4);
RP = features(:,5);
LGRE = features(:,6);
HGRE = features(:,7);
SRLGE = features(:,8);
SRHGE = features(:,9);
LRLGE = features(:,10);
LRHGE = features(:,11);
