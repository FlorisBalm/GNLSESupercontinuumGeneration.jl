# This is adapted from the lecture 'Introduction to NLSE simulation / supercontinuum generation'
# on youtube, by Miles Anderson.
# The code and parameters are those of Silicon Nitride

using DifferentialEquations
using Trapz
using Random
using Printf
using FFTW
using Plots


function propstep2(A,L,y,h)
    N = 1im*y*abs.(A).^2;
    B = fft(exp.(h.*L).*ifft(exp.(h.*N).*A))
    return B
end

function splitstep_run()
    npoints=2^9; 
    timewidth=5.0; # ps pulse width
    c=299792458*1e9/1e12; # nm/ps speed of light
    wavelength=1550 # nm wavelength of pulse centre

    ω0 = 2*π*c/wavelength;
    fasttime = collect(LinRange(-timewidth/2, timewidth/2, npoints));
    hbar = 6.636e-10/(2*π);
    repetitionrate = 100e6*1e-12;
    dT = fasttime[2]-fasttime[1];
    frequencies = 2*π/(npoints*dT)*collect(-(npoints>>1):(npoints>>1)-1) .+ ω0;

    #Random.seed!(0);
    # Add some noise to allow for noise-activated processes
    noiseterm = sqrt.(0im.+hbar*frequencies/(2*π*repetitionrate)).*exp.(-1im*2*π*rand(Float64,npoints))
    

    # wave guide parameters
    # This contains only β2, but can be expanded with higher order
    # coefficients 
    betas= [-0.13208];
    # betas = [-0.13208, 0.00045]
    n2 = 2.4e-19; # Nonlinear refractive index
    Aeff=1.32e-12; # Effective waveguie area
    gamma = ω0*n2/(c*1e-9*Aeff);
    loss = 0; # Turn off loss for now
    
    pulseduration = 0.150/1.7632; # pulse duration is 150fs
    Ldispersion=pulseduration^2/abs(betas[1]) 
    LS = Ldispersion*π/2;
    nsolitons=3; # Number of solitons
    Lnonlinear=Ldispersion/nsolitons^2;
    power = 1/(gamma*Lnonlinear);
    energy=2*power*pulseduration;

    pulseenvelope = sqrt(power)*sech.(fasttime./pulseduration);
    wglength = LS*2;
    stepsize = Lnonlinear/50; # Sets number of steps
    nsteps = floor(Int, wglength/stepsize);
    zcoord = LinRange(0.0,wglength,nsteps);
    ramanfraction = 0.0; #0.18 for glass

    tau1 = 0.0122; tau2 = 0.032;
    ramanterm = (tau1^2+tau2^2)/(tau1*tau2^2) .* exp.(-fasttime./tau2).*sin.(fasttime./tau1);

    ramanterm[fasttime .< 0] .= 0;
    ramanterm = ramanterm/(trapz(fasttime,ramanterm))
    
    fullenvelope = pulseenvelope .+ noiseterm;

    D = zeros(ComplexF64, length(frequencies));
    for i = 1:length(betas)
        D = D .+ betas[i]/factorial(i+1).*(frequencies .- ω0).^(i+1);
    end

    linearoperator = D.*1im .- loss/2;
    linearoperator = fftshift(linearoperator)

    resulttimedomain = zeros(ComplexF64,nsteps,npoints);
    for i=1:nsteps
        
        fullenvelope = propstep2(fullenvelope,linearoperator,gamma,stepsize)
        if (i % 100 == 0)
            @printf "%.3f steps completed" 100*i/nsteps
        end

        resulttimedomain[i,:] = fullenvelope;
    end



    
    resultfrequencydomain =  ifft(resulttimedomain, 2)
    resultfrequencydomain = ifftshift(resultfrequencydomain, 2)
    logintensitytime = 10*log10.(abs.(resultfrequencydomain).^2 .*timewidth .* repetitionrate) .+ 30;
    logintensityfrequency = 10*log10.(abs.(resulttimedomain).^2)

    wavelengthsplot = 2*π*c./frequencies;
    wavenumbers = wavelengthsplot./(2*π)

    XLi = [-150 150].+(ω0/(2*pi) )
    iif = (wavenumbers .> XLi[1] .&& wavenumbers .< XLi[2]);
    p1 = heatmap(logintensityfrequency);
    p2 = heatmap(logintensitytime);


    plot(p1, p2, layout=2);



end


