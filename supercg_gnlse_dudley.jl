# This is adapted from the lecture 'Introduction to NLSE simulation / supercontinuum generation'
# on youtube, by Miles Anderson.
# The code and parameters are those of Silicon Nitride
# The gnsle algorithm has been taken from  J.C. Travers, M.H. Frosz and J.M. Dudley in Chapter 3 of the book "Supercontinuum Generation in Optical Fibers" Edited by J. M. Dudley and J. R. Taylor (Cambridge 2010),
# which has also been publised on github by J.C. Travers on https://github.com/jtravs/SCGBookCode

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using DifferentialEquations
using Trapz
using Random
using FFTW
using Printf
using Plots

function gnlse(T, A, w0, gamma, betas, 
    loss, fr, RT, flength, nsaves)
    n = length(T); dT = T[2]-T[1]; # grid parameters
    V = 2*π.*collect(-n/2:n/2-1)./(n*dT); # frequency grid
    alpha = log(10^(loss/10));    # attenuation coefficient
    
    B = zeros(Float64,length(V));
    for i = 1:length(betas)        # Taylor expansion of betas
      B = B + betas[i]/factorial(i+1).*V.^(i+1);
    end
    L = 1im.*B .- alpha/2;            # linear operator

    if abs(w0) > eps()               # if w0>0 then include shock
        gamma = gamma/w0;    
        W = V .+ w0;                # for shock W is true freq
    else
        W = ones(length(V));                     # set W to 1 when no shock
    end

    RW = n.*ifft(fftshift(RT));   # frequency domain Raman
    L = fftshift(L); W = fftshift(W); # shift to fft space
    
    # The equation dz = rhs(z) is the first-order ode we wish to integrate
    function rhs!(dz, AW, p, z)
      AT = fft(AW.*exp.(L.*z));         # time domain field
      IT = abs.(AT).^2;                # tme domain intensity
      if (length(RT) == 1) || (abs(fr) < eps()) # no Raman case
        M = ifft(AT.*IT);             # response function
      else
        RS = dT.*fr.*fft(ifft(IT).*RW); # Raman convolution
        M = ifft(AT.*((1-fr).*IT .+ RS));# response function
      end
      dz[:] = 1im.*gamma.*W.*M.*exp.(-L.*z);   # full RHS of Eq. (3.13)
    end

    # the points to output the solution at
    Z = collect(LinRange(0, flength, nsaves));  

    # Defines the problem in the generic Julia sense of 
    # function - initial value - solution domain
    problem = ODEProblem(rhs!, ifft(A), [Z[1] Z[end]])

    # Progress creates a terminal progress indicator, which s very useful
    sol = solve(problem,Tsit5(), reltol=1e-7, abstol=1e-12,progress=true)

    # Output 
    AW = zeros(ComplexF64, nsaves, length(A))
    for i = 1:nsaves
        AW[i,:] = sol(Z[i])
    end
    
    AT = zeros(ComplexF64, size(AW));
    for i = 1:size(AW,1)
      AW[i,:] = AW[i,:].*exp.(L.*Z[i]); # change variables
      AT[i,:] = fft(AW[i,:]);           # time domain output
      AW[i,:] = fftshift(AW[i,:]).*dT.*n;  # scale
    end
    
    W = V .+ w0; # the absolute frequency grid

    return Z, AT, AW, W

end

function gnlse_run()
    npoints=2^10; 
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
    nsaves = 300;
    zcoord, soltimedomain, solfreqdomain, frequencies = gnlse(fasttime, pulseenvelope+noiseterm, 0, gamma, betas, loss, ramanfraction, ramanterm,wglength, nsaves )

    logintensityfrequency = 10*log10.(abs.(solfreqdomain).^2 .*timewidth .* repetitionrate) .+ 30;
    logintensitytime = 10*log10.(abs.(soltimedomain).^2)

    wavelenghts = 2*π*c./frequencies;
    F  = frequencies./(2*π)

    XLi = [-150 150].+(ω0/(2*π) )

    iif = (F .> XLi[1] .&& F .< XLi[2]);

    p1 = heatmap(logintensityfrequency);
    p2 = heatmap(logintensitytime);

    Plots.plot(p1, p2, layout=2);
end
