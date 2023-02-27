projectDir = "C:\Users\sande\Desktop\school\Senior Spring\Math and Music\InstrumentComparison\Recordings";

%% Create audio Datastore object
ads = audioDatastore(fullfile(projectDir, {'Wonderphone/Notes', 'Jupiter/Notes'}), ...
    'IncludeSubfolders', true, 'FileExtensions', '.wav');
fileNames = ads.Files;

%TPT_IDs = extractBetween(fileNames,"Recordings\","Notes");
%ads.Labels = categorical(TPT_IDs);

% Extract labels from filenames and add them to the datastore
for i = 1:numel(ads.Files)
    [~, filename, ~] = fileparts(ads.Files{i});
    parentdir = extractAfter(fileparts(ads.Files{i}), 'Recordings\');
    parentdir = extractBefore(parentdir, '\');
    labels{i} = parentdir;
end

% Convert labels to categorical array
uniqueLabels = unique(labels);
categoricalLabels = categorical(labels, uniqueLabels);

% Assign categorical labels to datastore
ads.Labels = categoricalLabels;

countEachLabel(ads)

%% Split data into sets
% Split the datastore into training, evaluation, and test sets
[adsTrain, adsEval, adsTest] = splitEachLabel(ads, 0.6, 0.2, 0.2, 'randomized');

%% Example file
[audio,audioInfo] = read(adsTrain);
fs = audioInfo.SampleRate;

t = (0:size(audio,1)-1)/fs;
sound(audio,fs)
plot(t,audio)
xlabel("Time (s)")
ylabel("Amplitude")
axis([0 t(end) -1 1])
title("Sample Utterance from Training Set")
reset(adsTrain)


%% Setting Up Feature Extraction

speedUpExample = false;
if speedUpExample
    adsTrain = splitEachLabel(adsTrain,30);
    adsDET = splitEachLabel(adsDET,21);
end

numCoeffs = 20;
deltaWindowLength = 9;
windowDuration = 0.025;
hopDuration = 0.01;

windowSamples = round(windowDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = windowSamples - hopSamples;

afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    Window=hann(windowSamples,"periodic"), ...
    OverlapLength=overlapSamples, ...
    ...
    mfcc=true, ...
    mfccDelta=true, ...
    mfccDeltaDelta=true);
setExtractorParameters(afe,"mfcc",DeltaWindowLength=deltaWindowLength,NumCoeffs=numCoeffs)

%% Extracting Features
features = extract(afe,audio);
[numHops,numFeatures] = size(features)

%% Train

if ~isempty(ver("parallel")) && ~speedUpExample
    pool = gcp;
    numPar = numpartitions(adsTrain,pool);
else
    numPar = 1;
end

%%
featuresAll = {};
tic
parfor ii = 1:numPar
    adsPart = partition(adsTrain,numPar,ii);
    featuresPart = cell(0,numel(adsPart.Files));
    for iii = 1:numel(adsPart.Files)
        audioData = read(adsPart);
        audioData = reshape(audioData,[],1); % Convert to column vector
        featuresPart{iii} = helperFeatureExtraction(audioData,afe,[]);
    end
    featuresAll = [featuresAll,featuresPart];
end
allFeatures = cat(2,featuresAll{:});
disp("Feature extraction from training set complete (" + toc + " seconds).")

%%
normFactors.Mean = mean(allFeatures,2,"omitnan");
normFactors.STD = std(allFeatures,[],2,"omitnan");

%% Universal Background Model (UBM)
% Initialize the Gaussian mixture model (GMM) that will be the universal background model (UBM) in the i-vector system. The component weights are initialized as evenly distributed. Systems trained on the TIMIT data set usually contain around 2048 components. 
numComponents = 64;
if speedUpExample
    numComponents = 32;
end
alpha = ones(1,numComponents)/numComponents;
mu = randn(numFeatures,numComponents);
vari = rand(numFeatures,numComponents) + eps;
ubm = struct(ComponentProportion=alpha,mu=mu,sigma=vari);

%% Train the UBM using the expectation-maximization (EM) algorithm.
maxIter = 25;
if speedUpExample
    maxIter = 2;
end
tic
for iter = 1:maxIter
    tic
    % EXPECTATION
    N = zeros(1,numComponents);
    F = zeros(numFeatures,numComponents);
    S = zeros(numFeatures,numComponents);
    L = 0;
    parfor ii = 1:numPar
        adsPart = partition(adsTrain,numPar,ii);
        while hasdata(adsPart)
            audioData = read(adsPart);
            audioData = reshape(audioData,[],1); % Convert to column vector

            % Extract features
            Y = helperFeatureExtraction(audioData,afe,normFactors);
 
            % Compute a posteriori log-liklihood
            logLikelihood = helperGMMLogLikelihood(Y,ubm);

            % Compute a posteriori normalized probability
            amax = max(logLikelihood,[],1);
            logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
            gamma = exp(logLikelihood - logLikelihoodSum)';
            
            % Compute Baum-Welch statistics
            n = sum(gamma,1);
            f = Y * gamma;
            s = (Y.*Y) * gamma;
            
            % Update the sufficient statistics over utterances
            N = N + n;
            F = F + f;
            S = S + s;
            
            % Update the log-likelihood
            L = L + sum(logLikelihoodSum);
        end
    end
    
    % Print current log-likelihood
    disp("Training UBM: " + iter + "/" + maxIter + " complete (" + round(toc) + " seconds), Log-likelihood = " + round(L))
    
    % MAXIMIZATION
    N = max(N,eps);
    ubm.ComponentProportion = max(N/sum(N),eps);
    ubm.ComponentProportion = ubm.ComponentProportion/sum(ubm.ComponentProportion);
    ubm.mu = F./N;
    ubm.sigma = max(S./N - ubm.mu.^2,eps);
end

%% Calculate Baum-Welch Statistics
numSpeakers = numel(adsTrain.Files);
Nc = {};
Fc = {};

tic
parfor ii = 1:numPar
    adsPart = partition(adsTrain,numPar,ii);
    numFiles = numel(adsPart.Files);
    
    Npart = cell(1,numFiles);
    Fpart = cell(1,numFiles);
    for jj = 1:numFiles
        audioData = read(adsPart);
        audioData = reshape(audioData,[],1); % Convert to column vector

        % Extract features
        Y = helperFeatureExtraction(audioData,afe,normFactors);
        
        % Compute a posteriori log-likelihood
        logLikelihood = helperGMMLogLikelihood(Y,ubm);
        
        % Compute a posteriori normalized probability
        amax = max(logLikelihood,[],1);
        logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
        gamma = exp(logLikelihood - logLikelihoodSum)';
        
        % Compute Baum-Welch statistics
        n = sum(gamma,1);
        f = Y * gamma;
        
        Npart{jj} = reshape(n,1,1,numComponents);
        Fpart{jj} = reshape(f,numFeatures,1,numComponents);
    end
    Nc = [Nc,Npart];
    Fc = [Fc,Fpart];
end
disp("Baum-Welch statistics completed (" + toc + " seconds).")

%% Expand the statistics into matrices and center , as described in [3], such that
%is a  diagonal matrix whose blocks are .
%is a  supervector obtained by concatenating .
%is the number of components in the UBM.
%is the number of features in a feature vector.

N = Nc;
F = Fc;
muc = reshape(ubm.mu,numFeatures,1,[]);
for s = 1:numSpeakers
    N{s} = repelem(reshape(Nc{s},1,[]),numFeatures);
    F{s} = reshape(Fc{s} - Nc{s}.*muc,[],1);
end

%% Total Variability Space
Sigma = ubm.sigma(:);
numTdim = 32;
if speedUpExample
    numTdim = 16;
end

T = randn(numel(ubm.sigma),numTdim);
T = T/norm(T);

I = eye(numTdim);

Ey = cell(numSpeakers,1);
Eyy = cell(numSpeakers,1);
Linv = cell(numSpeakers,1);

%Set the number of iterations for training. A typical value reported is 20.
numIterations = 20;

%% Run the training loop.
for iterIdx = 1:numIterations
    tic
    
    % 1. Calculate the posterior distribution of the hidden variable
    TtimesInverseSSdiag = (T./Sigma)';
    parfor s = 1:numSpeakers
        L = (I + TtimesInverseSSdiag.*N{s}*T);
        Linv{s} = pinv(L);
        Ey{s} = Linv{s}*TtimesInverseSSdiag*F{s};
        Eyy{s} = Linv{s} + Ey{s}*Ey{s}';
    end
    
    % 2. Accumlate statistics across the speakers
    Eymat = cat(2,Ey{:});
    FFmat = cat(2,F{:});
    Kt = FFmat*Eymat';
    K = mat2cell(Kt',numTdim,repelem(numFeatures,numComponents));
    
    newT = cell(numComponents,1);
    for c = 1:numComponents
        AcLocal = zeros(numTdim);
        for s = 1:numSpeakers
            AcLocal = AcLocal + Nc{s}(:,:,c)*Eyy{s};
        end
        
    % 3. Update the Total Variability Space
        newT{c} = (pinv(AcLocal)*K{c})';
    end
    T = cat(1,newT{:});

    disp("Training Total Variability Space: " + iterIdx + "/" + numIterations + " complete (" + round(toc,2) + " seconds).")
end

%% i-Vector Extraction
speakers = unique(adsTrain.Labels);
numSpeakers = numel(speakers);
ivectorPerSpeaker = cell(numSpeakers,1);
TS = T./Sigma;
TSi = TS';
ubmMu = ubm.mu;
tic
parfor speakerIdx = 1:numSpeakers
    
    % Subset the datastore to the speaker you are adapting.
    adsPart = subset(adsTrain,adsTrain.Labels==speakers(speakerIdx));
    numFiles = numel(adsPart.Files);
    
    ivectorPerFile = zeros(numTdim,numFiles);
    for fileIdx = 1:numFiles
        audioData = read(adsPart);
        audioData = reshape(audioData,[],1); % Convert to column vector
        
        % Extract features
        Y = helperFeatureExtraction(audioData,afe,normFactors);
        
        % Compute a posteriori log-likelihood
        logLikelihood = helperGMMLogLikelihood(Y,ubm);
        
        % Compute a posteriori normalized probability
        amax = max(logLikelihood,[],1);
        logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
        gamma = exp(logLikelihood - logLikelihoodSum)';
        
        % Compute Baum-Welch statistics
        n = sum(gamma,1);
        f = Y * gamma - n.*(ubmMu);

        ivectorPerFile(:,fileIdx) = pinv(I + (TS.*repelem(n(:),numFeatures))' * T) * TSi * f(:);
    end
    ivectorPerSpeaker{speakerIdx} = ivectorPerFile;
end
disp("I-vectors extracted from training set (" + toc + " seconds).")

%% Projection Matrix

w = ivectorPerSpeaker;
utterancePerSpeaker = cellfun(@(x)size(x,2),w);

ivectorsTrain = cat(2,w{:});
projectionMatrix = eye(size(w{1},1));

performLDA = true;
if performLDA
    tic
    numEigenvectors = 16;

    Sw = zeros(size(projectionMatrix,1));
    Sb = zeros(size(projectionMatrix,1));
    wbar = mean(cat(2,w{:}),2);
    for ii = 1:numel(w)
        ws = w{ii};
        wsbar = mean(ws,2);
        Sb = Sb + (wsbar - wbar)*(wsbar - wbar)';
        Sw = Sw + cov(ws',1);
    end
    
    [A,~] = eigs(Sb,Sw,numEigenvectors);
    A = (A./vecnorm(A))';

    ivectorsTrain = A * ivectorsTrain;
    
    w = mat2cell(ivectorsTrain,size(ivectorsTrain,1),utterancePerSpeaker);
    
    projectionMatrix = A * projectionMatrix;
    
    disp("LDA projection matrix calculated (" + round(toc,2) + " seconds).")
end

%% WCCN

performWCCN = true;
if performWCCN
    tic
    alpha = 0.9;
    
    W = zeros(size(projectionMatrix,1));
    for ii = 1:numel(w)
        W = W + cov(w{ii}',1);
    end
    W = W/numel(w);
    
    W = (1 - alpha)*W + alpha*eye(size(W,1));
    
    B = chol(pinv(W),"lower");
    
    projectionMatrix = B * projectionMatrix;
    
    disp("WCCN projection matrix calculated (" + round(toc,4) + " seconds).")
end





%% Train G-PLDA Model
%
%
%
%
%
% Apply the projection matrix to the train set.
ivectors = cellfun(@(x)projectionMatrix*x,ivectorPerSpeaker,UniformOutput=false);

numEigenVoices = 16;
K = numel(ivectors);
D = size(ivectors{1},1);
utterancePerSpeaker = cellfun(@(x)size(x,2),ivectors);

ivectorsMatrix = cat(2,ivectors{:});
N = size(ivectorsMatrix,2);
mu = mean(ivectorsMatrix,2);

ivectorsMatrix = ivectorsMatrix - mu;

whiteningType = 'ZCA';

if strcmpi(whiteningType,"ZCA")
    S = cov(ivectorsMatrix');
    [~,sD,sV] = svd(S);
    W = diag(1./(sqrt(diag(sD)) + eps))*sV';
    ivectorsMatrix = W * ivectorsMatrix;
elseif strcmpi(whiteningType,"PCA")
    S = cov(ivectorsMatrix');
    [sV,sD] = eig(S);
    W = diag(1./(sqrt(diag(sD)) + eps))*sV';
    ivectorsMatrix = W * ivectorsMatrix;
else
    W = eye(size(ivectorsMatrix,1));
end

ivectorsMatrix = ivectorsMatrix./vecnorm(ivectorsMatrix);
S = ivectorsMatrix*ivectorsMatrix';
ivectors = mat2cell(ivectorsMatrix,D,utterancePerSpeaker);

uniqueLengths = unique(utterancePerSpeaker);
numUniqueLengths = numel(uniqueLengths);

speakerIdx = 1;
f = zeros(D,K);
for uniqueLengthIdx = 1:numUniqueLengths
    idx = find(utterancePerSpeaker==uniqueLengths(uniqueLengthIdx));
    temp = {};
    for speakerIdxWithinUniqueLength = 1:numel(idx)
        rho = ivectors(idx(speakerIdxWithinUniqueLength));
        temp = [temp;rho]; %#ok<AGROW>

        f(:,speakerIdx) = sum(rho{:},2);
        speakerIdx = speakerIdx+1;
    end
    ivectorsSorted{uniqueLengthIdx} = temp; %#ok<SAGROW> 
end

V = randn(D,numEigenVoices);
Lambda = pinv(S/N);

numIter = 10;
minimumDivergence = true;

%% Train the G-PLDA model using the EM algorithm

for iter = 1:numIter
    % EXPECTATION
    gamma = zeros(numEigenVoices,numEigenVoices);
    EyTotal = zeros(numEigenVoices,K);
    R = zeros(numEigenVoices,numEigenVoices);
    
    idx = 1;
    for lengthIndex = 1:numUniqueLengths
        ivectorLength = uniqueLengths(lengthIndex);
        
        % Isolate i-vectors of the same given length
        iv = ivectorsSorted{lengthIndex};
        
        % Calculate M
        M = pinv(ivectorLength*(V'*(Lambda*V)) + eye(numEigenVoices)); % Equation (A.7) in [13]
        
        % Loop over each speaker for the current i-vector length
        for speakerIndex = 1:numel(iv)
            
            % First moment of latent variable for V
            Ey = M*V'*Lambda*f(:,idx); % Equation (A.8) in [13]
            
            % Calculate second moment.
            Eyy = Ey * Ey';
            
            % Update Ryy 
            R = R + ivectorLength*(M + Eyy); % Equation (A.13) in [13]
            
            % Append EyTotal
            EyTotal(:,idx) = Ey;
            idx = idx + 1;
            
            % If using minimum divergence, update gamma.
            if minimumDivergence
                gamma = gamma + (M + Eyy); % Equation (A.18) in [13]
            end
        end
    end
    
    % Calculate T
    TT = EyTotal*f'; % Equation (A.12) in [13]
    
    % MAXIMIZATION
    V = TT'*pinv(R); % Equation (A.16) in [13]
    Lambda = pinv((S - V*TT)/N); % Equation (A.17) in [13]

    % MINIMUM DIVERGENCE
    if minimumDivergence
        gamma = gamma/K; % Equation (A.18) in [13]
        V = V*chol(gamma,'lower');% Equation (A.22) in [13]
    end
end

%%
speakerIdx = 2;
utteranceIdx = 1;
w1 = ivectors{speakerIdx}(:,utteranceIdx);

speakerIdx = 1;
utteranceIdx = 10;
wt = ivectors{speakerIdx}(:,utteranceIdx);

VVt = V*V';
SigmaPlusVVt = pinv(Lambda) + VVt;

term1 = pinv([SigmaPlusVVt VVt;VVt SigmaPlusVVt]);
term2 = pinv(SigmaPlusVVt);

w1wt = [w1;wt];
score = w1wt'*term1*w1wt - w1'*term2*w1 - wt'*term2*wt

%%
gpldaModel = struct(mu=mu, ...
                    WhiteningMatrix=W, ...
                    EigenVoices=V, ...
                    Sigma=pinv(Lambda));

%% Helper functions
%Feature Extraction and Normalization
function [features,numFrames] = helperFeatureExtraction(audioData,afe,normFactors)
    % Input:
    % audioData   - column vector of audio data
    % afe         - audioFeatureExtractor object
    % normFactors - mean and standard deviation of the features used for normalization. 
    %               If normFactors is empty, no normalization is applied.
    %
    % Output
    % features    - matrix of features extracted
    % numFrames   - number of frames (feature vectors) returned
    
    % Normalize
    audioData = audioData/max(abs(audioData(:)));
    
    % Protect against NaNs
    audioData(isnan(audioData)) = 0;
    
    % Isolate speech segment
    idx = detectSpeech(audioData,afe.SampleRate);
    features = [];
    for ii = 1:size(idx,1)
        f = extract(afe,audioData(idx(ii,1):idx(ii,2)));
        features = [features;f]; %#ok<AGROW> 
    end

    % Feature normalization
    if ~isempty(normFactors)
        features = (features-normFactors.Mean')./normFactors.STD';
    end
    features = features';
    
    % Cepstral mean subtraction (for channel noise)
    if ~isempty(normFactors)
        features = features - mean(features,"all");
    end
    
    numFrames = size(features,2);
end
%% Gaussian Multi-Component Mixture Log-Likelihood
function L = helperGMMLogLikelihood(x,gmm)
    xMinusMu = repmat(x,1,1,numel(gmm.ComponentProportion)) - permute(gmm.mu,[1,3,2]);
    permuteSigma = permute(gmm.sigma,[1,3,2]);
    
    Lunweighted = -0.5*(sum(log(permuteSigma),1) + sum(xMinusMu.*(xMinusMu./permuteSigma),1) + size(gmm.mu,1)*log(2*pi));
    
    temp = squeeze(permute(Lunweighted,[1,3,2]));
    if size(temp,1)==1
        % If there is only one frame, the trailing singleton dimension was
        % removed in the permute. This accounts for that edge case.
        temp = temp';
    end
    
    L = temp + log(gmm.ComponentProportion)';
end
%% G-PLDA Score
function score = gpldaScore(gpldaModel,w1,wt)
% Center the data
w1 = w1 - gpldaModel.mu;
wt = wt - gpldaModel.mu;

% Whiten the data
w1 = gpldaModel.WhiteningMatrix*w1;
wt = gpldaModel.WhiteningMatrix*wt;

% Length-normalize the data
w1 = w1./vecnorm(w1);
wt = wt./vecnorm(wt);

% Score the similarity of the i-vectors based on the log-likelihood.
VVt = gpldaModel.EigenVoices * gpldaModel.EigenVoices';
SVVt = gpldaModel.Sigma + VVt;

term1 = pinv([SVVt VVt;VVt SVVt]);
term2 = pinv(SVVt);

w1wt = [w1;wt];
score = w1wt'*term1*w1wt - w1'*term2*w1 - wt'*term2*wt;
end

