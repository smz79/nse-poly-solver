clear variables 
close all

%% Settings 
usePseudoColor=1;
savePlotsAndRecon=0;

%% FORWARD configuration 

scenario = 'hi-res';  
%scenario = 'low-res';  
% scenario = 'mid-res'; 

switch(scenario)
    
    case 'hi-res'
        % Forward is numerically stable for very hi-res (~500), low visc
        % (~0.001) provided Delta_t=0.0005 but filter requires to further
        % reduce the timestep, e.g. Delta_t=0.0005/6
        Delta_t             = 0.0005/3; %
        endTime             = 30;     %final time 
        N_steps             = endTime/Delta_t;
        
        Nx                  = 500;    %# of basis functions in x-dimension
        Ny                  = Nx;     %# of basis functions in y-dimension
        
        alpha               = 0.00;   %damping 
        nu                  = 0.0009; %diffusion
        
        
        solver              = 'midpoint';
        midPointIterations  =  5;
        
        forcingMode         =  6; %forcing will be applied at |n|=forcingMode (6 is good)
        forcingAmplitude    =  1;
        Kolmogorov_forcing            =  forcing( forcingMode, forcingAmplitude, Nx, Ny );

        initialProfile = @(x,y)3*(1-(x-pi)).^2.*exp(-((x-pi).^2) - ((y-pi)+1).^2) ...
            - 10*((x-pi)/5 - (x-pi).^3 - (y-pi).^5).*exp(-(x-pi).^2-(y-pi).^2) ... 
            - 1/3*exp(-((x-pi)+1).^2 - (y-pi).^2); 
        
        saveInterval        =  0.05; 
        displayInterval     =  saveInterval;
        
        Observer.L = min(1e4,10/nu); 
        Observer.dx = 2*pi/15; 
        Observer.dy = 2*pi/15; 
    
    case 'mid-res'
        %For k-forcing: e.g. Nx=Ny=200; nu=0.01; alpha=0; IC=peaks; forcemode=6;
        %forceamp=1; dt=0.005; bounds +/-8 from min/max of IC; midPointIter=5;

        Delta_t  = 0.0009; %Delta_t=0.0005 for very hi-res (~500), low visc (~0.001)
        endTime  = 10.0005;    %final time 
        N_steps  = endTime/Delta_t;
        
        Nx       = 300;    %# of basis functions in x-dimension
        Ny       = Nx;    %# of basis functions in y-dimension
        
        alpha    = 0.00;  %damping 
        nu       = 0.009;  %diffusion
        
        solver   = 'midpoint';
        midPointIterations=5;
        
        forcingMode         =  6; %forcing will be applied at |n|=forcingMode (6 is good)
        forcingAmplitude    =  1;
        Kolmogorov_forcing            =  forcing( forcingMode, forcingAmplitude, Nx, Ny );

        initialProfile = @(x,y)3*(1-(x-pi)).^2.*exp(-((x-pi).^2) - ((y-pi)+1).^2) ...
            - 10*((x-pi)/5 - (x-pi).^3 - (y-pi).^5).*exp(-(x-pi).^2-(y-pi).^2) ... 
            - 1/3*exp(-((x-pi)+1).^2 - (y-pi).^2); 
        
        saveInterval        =  0.01; 
        displayInterval     =  saveInterval;
        
        Observer.L = 10/nu; 
        Observer.dx = 2*pi/40; 
        Observer.dy = 2*pi/40; 


        
    case 'low-res'
        %For k-forcing: e.g. Nx=Ny=100; nu=0.01; alpha=0; IC=peaks; forcemode=6;
        %forceamp=1; dt=0.01; bounds +/-8 from min/max of IC; midPointIter=5; 

        Delta_t  = 0.01/2; %Delta_t=0.0005 for very hi-res (~500), low visc (~0.001)
        endTime  = 20;    %final time 
        N_steps  = endTime/Delta_t;
        
        Nx       = 100;    %# of basis functions in x-dimension
        Ny       = Nx;    %# of basis functions in y-dimension
        
        alpha    = 0.00;  %damping 
        nu       = 0.05;  %diffusion
        
        solver   = 'midpoint';
        midPointIterations=5;
        
        forcingMode         =  6; %forcing will be applied at |n|=forcingMode (6 is good)
        forcingAmplitude    =  1;
        Kolmogorov_forcing            =  forcing( forcingMode, forcingAmplitude, Nx, Ny );

        initialProfile = @(x,y)3*(1-(x-pi)).^2.*exp(-((x-pi).^2) - ((y-pi)+1).^2) ...
            - 10*((x-pi)/5 - (x-pi).^3 - (y-pi).^5).*exp(-(x-pi).^2-(y-pi).^2) ... 
            - 1/3*exp(-((x-pi)+1).^2 - (y-pi).^2); 
        
        saveInterval        =  0.01; 
        displayInterval     =  saveInterval;
        
        Observer.L = 10/nu; 
        Observer.dx = 2*pi/40; 
        Observer.dy = 2*pi/40; 

       
    otherwise
        
        warning('Using low resolution scenario!'); 
        Delta_t  = 0.05; %Delta_t=0.0005 for very hi-res (~500), low visc (~0.001)
        endTime  = 20;    %final time 
        Nx       = 10;    %# of basis functions in x-dimension
        Ny       = Nx;    %# of basis functions in y-dimension
        alpha    = 0.00;  %damping 
        nu       = 0.01;  %diffusion

        N_steps  = endTime/Delta_t;
        solver   = 'midpoint';
        midPointIterations=5;
        forcingMode         =  6; %forcing will be applied at |n|=forcingMode (6 is good)
        forcingAmplitude    =  1;
        Kolmogorov_forcing            =  forcing( forcingMode, forcingAmplitude, Nx, Ny );

        initialProfile = @(x,y)3*(1-(x-pi)).^2.*exp(-((x-pi).^2) - ((y-pi)+1).^2) ...
            - 10*((x-pi)/5 - (x-pi).^3 - (y-pi).^5).*exp(-(x-pi).^2-(y-pi).^2) ... 
            - 1/3*exp(-((x-pi)+1).^2 - (y-pi).^2); 
        saveInterval        =  0.05; 
        displayInterval     =  saveInterval;


end

%grid
leftBoundaryX=0; %don't change these
leftBoundaryY=0;
rightBoundaryX=2*pi;
rightBoundaryY=2*pi;
integrationGridX=linspace(leftBoundaryX,rightBoundaryX,Nx+1);
integrationGridY=linspace(leftBoundaryY,rightBoundaryY,Ny+1);
Lx = rightBoundaryX - leftBoundaryX; 
Ly = rightBoundaryY - leftBoundaryY;

[Xplot,Yplot] = meshgrid(integrationGridX,integrationGridY);

integrationGridX(end)=[];
integrationGridY(end)=[];

[X,Y]=meshgrid(integrationGridX,integrationGridY);

Delta_x=integrationGridX(2)-integrationGridX(1);
Delta_y=integrationGridY(2)-integrationGridY(1);




%% Initial vorticity
f=initialProfile(X,Y);
%f = f./max(abs(f(:)))+sin(randi([40,50],size(X)).*X).*sin(randi([40,50],size(X)).*Y); 
f = f./max(abs(f(:)))+sin(randi([29,33],size(X)).*X).*sin(randi([29,33],size(X)).*Y); 


pc=fft2(f);
pc(1,1)=0; 
recon=real(ifft2(pc));
h=figure;
surf(X,Y,recon)
colourBounds=[min(min(recon))-8,max(max(recon))+8];
%colourBounds=[min(min(recon)),max(max(recon))];
axis([leftBoundaryX rightBoundaryX-Delta_x leftBoundaryY rightBoundaryY-Delta_y colourBounds(1) colourBounds(2)])
%caxis([colourBounds(1) colourBounds(2)])
title('t = 0.00')
fprintf('Relative reconstruction error = %.3f%% \n', norm(f(:)-recon(:))/norm(f(:))*100)
if usePseudoColor==1
    view(0,90);
    shading flat;
end

if savePlotsAndRecon==1
    saveas(h,'plot_0.00.jpg','jpg')
end
drawnow
%% Forward 

pc=fft2(f);

%vorticity_pc = FORWARD_NSEvort_pseudospectral_midpoint(pc, Kolmogorov_forcing, Nx, Ny, Delta_t, endTime, alpha, nu, midPointIterations ,saveInterval); 

pc(1,1) = 0; %remove mean 
pc=fftshift(pc);
pc=pc(:)/(Nx*Ny);

pc_est = 0*pc; 

vid=0;

%plotTimes=[0.5 1 2]; %enter times at which plots are required for saving
plotTimes=[];
plotNum=1;

State    = zeros(Nx*Ny,ceil(endTime/saveInterval)+1); %for storing data for filtering

for t=Delta_t:Delta_t:endTime
    
    switch solver
            
        case 'RK4'
            tic
            slope_acc=0;
            for k=1:4 
                if k==1
                    coeffs=pc;
                elseif k==2 
                    coeffs=pc + Delta_t/2*adot;
                    slope_acc=slope_acc+adot;
                elseif k==3
                    coeffs=pc + Delta_t/2*adot; 
                    slope_acc=slope_acc+adot*2;
                elseif k==4
                    coeffs=pc + Delta_t  *adot;
                    slope_acc=slope_acc+adot*2;
                end
               
                adot=zeros((Nx+1)*(Ny+1),1); 
                for i=1:Nx+1
                    for j=1:Ny+1
                        adot((Nx+1)*(i-1)+j,1)=eulerFG_ODE(i-Nx/2-1 ,j-Ny/2-1 ,Nx,Ny,coeffs,alpha,nu);
                    end
                end 
                
                adot = adot + Kolmogorov_forcing; 
                
            end
                
            slope_acc=slope_acc+adot;
            pc = pc + Delta_t*slope_acc/6;
                        
        case 'midpoint'
            %tic
                       
            if t==Delta_t
                [ind1,ind2]=meshgrid((-Nx/2:Nx/2-1),(-Ny/2:Ny/2-1));
                visc = -nu*(ind1.^2 + ind2.^2);
                diagonalA=visc(:)-alpha;
                diagonalMidpointMat=1-0.5*Delta_t*diagonalA;
                diagonalMidpointMatInv=diagonalMidpointMat.^-1;
                
                
                coeffs1 = 1i*ind2./(ind1.^2 + ind2.^2); %stuff for nl term
                coeffs2 = 1i*ind1;
                coeffs1(isnan(coeffs1))=0;
                coeffs3 = 1i*ind1./(ind1.^2 + ind2.^2);
                coeffs4 = 1i*ind2;
                coeffs3(isnan(coeffs3))=0;
            end
                                         
            
            pc_bar=pc;
            for i=1:midPointIterations
                nl_term=getFT_nl(pc_bar,Nx,Ny,coeffs1,coeffs2,coeffs3,coeffs4);
                pc_bar=diagonalMidpointMatInv.*(pc + Delta_t/2 * (Kolmogorov_forcing+nl_term));    
            end             
            pc=2*pc_bar-pc;
            
           
            
            
            otherwise
            
            error('Invalid solver!')
            
    end
    %toc;
    
    if savePlotsAndRecon==0
    
        if mod(t,displayInterval)==0
            vid=vid+1; 
            recon=real(ifft2(Nx*Ny*ifftshift(reshape(pc,Nx,[])))); 
            State(:,vid)=recon(:);
            
            recon_plot = sign(recon).*power(abs(recon)./max(abs(recon(:))),1/4);
            
            surf(X,Y,recon_plot);
            axis([leftBoundaryX rightBoundaryX-Delta_x leftBoundaryY rightBoundaryY-Delta_y [-1 1] [-1 1]])
            colorbar
            %
            titleText=sprintf('Truth: t = %.2f',t); 
            title(titleText);
            if usePseudoColor==1
                view(0,90);
                shading flat;
            end
            
            drawnow 
        end

    elseif savePlotsAndRecon==1
        
        if mod(t,displayInterval)==0
            recon=real(ifft2(Nx*Ny*ifftshift(reshape(pc,Nx,[]))));  
            if mod(t,saveInterval)==0
                close all
                h=figure;
            end
            recon_plot = recon./max(recon(:));
            surf(X,Y,recon_plot);
            axis([leftBoundaryX rightBoundaryX-Delta_x leftBoundaryY rightBoundaryY-Delta_y colourBounds(1) colourBounds(2)])
            %caxis([colourBounds(1) colourBounds(2)]);
            titleText=sprintf('t = %.2f',t);
            title(titleText);
            if usePseudoColor==1
                view(0,90);
                shading flat;
            end

            if plotNum<numel(plotTimes)+1 && abs(plotTimes(plotNum)-t)<Delta_t/2;
               pause 
               plotNum=plotNum+1;
            end

            if mod(t,saveInterval)==0
                %filename=sprintf('plot_%.2f.jpg',t);
                %saveas(h,filename,'jpg')

                filename2=sprintf('recon_%.2f.txt',t);
                save(filename2,'recon','-ascii')
            end
            vid=vid+1;
            mov(vid)=getframe;
        end
    end
    
end

save('NSE-500x500-nu0.0009-Obs20x20.mat');

%save obs 
est_err = State-Estimate;
L2norm_solution = sqrt(transpose(sum(State.*conj(State),1)));
L2norm_est_err = sqrt(transpose(sum(est_err.*conj(est_err),1)));
figure, semilogy(L2norm_est_err./L2norm_solution);
title('L2 rel error (log-scale)'); 

%save(['Truth_NxNy' num2str(max(Nx,Ny)) '_sampling' num2str(displayInterval) '_nu' num2str(nu) '_KF'],'Y_full')
%% plot
% v = VideoWriter('newfile.avi');
% open(v); 
% for jj=1:size(State,2)-1
%     recon = reshape(State(:,jj),Nx,Ny); 
%     recon_plot = sign(recon).*power(abs(recon)./max(abs(recon(:))),1/4);
%     subplot(1,2,1),
%     surf(X,Y,recon_plot);
%     axis([leftBoundaryX rightBoundaryX-Delta_x leftBoundaryY rightBoundaryY-Delta_y [-1 1] [-1 1]])
%     colorbar
%     %caxis([colourBounds(1) colourBounds(2)]);
%     titleText=sprintf('Random sin noise, re.err. = %.10f, t = %d',norm(tmp.Y_full(:,jj+1)-State(:,jj))/norm(State(:,jj)),jj);
%     title(titleText);
%     view(0,90);
%     shading flat; 
%     drawnow
%     recon = reshape(tmp.Y_full(:,jj+1),Nx,Ny); 
%     recon_plot = sign(recon).*power(abs(recon)./max(abs(recon(:))),1/4);
%     subplot(1,2,2),
%     surf(X,Y,recon_plot);
%     axis([leftBoundaryX rightBoundaryX-Delta_x leftBoundaryY rightBoundaryY-Delta_y [-1 1] [-1 1]])
%     colorbar
%     %caxis([colourBounds(1) colourBounds(2)]);
%     titleText=sprintf('Random noise, t = %.2f',jj);
%     title(titleText);
%     view(0,90);
%     shading flat; 
%     drawnow
%     frame = getframe(gcf);
%     writeVideo(v,frame);
%             
% end
% close(v);
