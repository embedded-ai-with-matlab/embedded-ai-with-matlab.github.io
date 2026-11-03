function ci = computeMotionIndicators(window) %#codegen
%COMPUTEMOTIONINDICATORS Condition indicators of a 6-axis IMU window.
%   ci = computeMotionIndicators(window) takes a 100-by-6 window
%   [ax ay az gx gy gz] (g and deg/s, 1 s at 100 Hz) and returns a 1-by-24
%   row vector with four indicators per channel:
%
%     1-6    log10 RMS        (vibration energy)
%     7-12   crest factor     (peak / RMS, high for impacts)
%     13-18  kurtosis         (high for short impulsive events)
%     19-24  log10 energy above 10 Hz relative to total (spectral shape)
%
%   The mean of each channel is removed first, so the board's orientation
%   (gravity) and the gyroscope's bias do not affect the indicators.

windowLength = 100;
x = single(window);
x = x - mean(x,1);

rmsValue = sqrt(mean(x.^2,1)) + single(1e-6);
crestFactor = max(abs(x),[],1) ./ rmsValue;
m2 = mean(x.^2,1) + single(1e-12);
kurt = mean(x.^4,1) ./ (m2.^2);

P = abs(fft(x)).^2;
total = sum(P(2:windowLength/2+1,:),1) + single(1e-12);
high = sum(P(12:windowLength/2+1,:),1) + single(1e-12);   % bins 11-50 Hz
highFraction = log10(high ./ total);

ci = [log10(rmsValue), crestFactor, kurt, highFraction];
end
