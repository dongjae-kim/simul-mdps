addpath('\\143.248.32.146\sjh\kdj\TerminalControl');
addpath('Z:\kdj\TerminalControl');
id = 'sjh';
pw = 'kinggodjh';
job.name = 'print_run.py';
job.path0 = '/home/sjh/kdj/parameter_recovery/gen_simulation_dat/gail/';
job.argu = [''];
job.pwd = job.path0;
job.nth = i;
JobPython(id,job,'old_torch',1,rem(i,3)+1);
[out] = SubmitJob(id,pw,job);


%%
% batch jobs for
% pause(60*60*5)
addpath('\\143.248.32.114\sjh\kdj\TerminalControl');

id = 'sjh';
pw = 'kinggodjh';
%         batch_information_recovery_subjective(i,ktov)
numsim = 2;
for i = 0:1:81
    %                 fl = dir(['simul_results/T' num2str(t) '/MODEL7/SUB' sprintf('%03d',i+1) '_SIMUL_BHV.npy']);
    %                 if size(fl,1)==0
    %             batch_simul_gen_by_regression(mod,i,sisi);
    job.name = 'run.py';
    job.path0 = '/home/sjh/kdj/parameter_recovery/gen_simulation_dat/gail/';
    job.argu = [sprintf('--subid=%d --numiter=%d',i,numsim)];
    job.pwd = job.path0;
    job.nth = i;
    JobPython(id,job,'old_torch',1,rem(i,3)+1);
    [out] = SubmitJob(id,pw,job);
end

%% generating simuls
addpath('\\143.248.30.101\sjh\kdj\TerminalControl');

id = 'sjh';
pw = 'kinggodjh';
%         batch_information_recovery_subjective(i,ktov)
numsim = 1;
for i = 0:1:81
    for numiter = 0:1:99
        
        fls = dir(['bhv_results' num2str(numsim+1) '/' num2str(numiter) '/SUB' sprintf('%03d_BHV.csv',i+1) ]);
        %                         fl = dir(['simul_results/T' num2str(t) '/MODEL7/SUB' sprintf('%03d',i+1) '_SIMUL_BHV.npy']);
        if size(fls,1)==0
            %             batch_simul_gen_by_regression(mod,i,sisi);
            job.name = 'generates.py';
            job.path0 = '/home/sjh/kdj/parameter_recovery/gen_simulation_dat/gail/';
            job.argu = [sprintf('--subid=%d --numsim=%d --numiter=%d',i,numiter,numsim)];
            job.pwd = job.path0;
            job.nth = i;
            JobPython(id,job,'old_torch',1,rem(i,3)+1);
            [out] = SubmitJob(id,pw,job);
        end
    end
end
%%
for numsim =2
    for i = 0:1:81
        fl = dir(['save_model/' num2str(numsim) '/' sprintf('SUB%03d_ckpt_*.pth.tar',i+1)]);
        fl2= dir(['save_model/' num2str(numsim) '/' sprintf('SUB%03d_ckpt_0.pth.tar',i+1)]);
        
        if size(fl2,1) ~= 0
            copyfile([fl(end).folder '/' fl(end).name],['save_model/' num2str(numsim) '/' sprintf('SUB%03d_ckpt_0.pth.tar',i+1)])
        end
        
        fl2= dir(['save_model/' num2str(numsim) '/' sprintf('SUB%03d_ckpt_0.pth.tar',i+1)]);
        
        if size(fl2,1)==0
            %             batch_simul_gen_by_regression(mod,i,sisi);
            job.name = 'run.py';
            job.path0 = '/home/sjh/kdj/parameter_recovery/gen_simulation_dat/gail/';
            job.argu = [sprintf('--subid=%d --numiter=%d',i,numsim)];
            job.pwd = job.path0;
            job.nth = i;
            JobPython(id,job,'old_torch',1,rem(i,3)+1);
            [out] = SubmitJob(id,pw,job);
        end
    end
    
    pause(120*60)
    addpath('\\143.248.32.114\sjh\kdj\TerminalControl');
    
    id = 'sjh';
    pw = 'kinggodjh';
    %         batch_information_recovery_subjective(i,ktov)
    for i = 0:1:81
        for numiter = 0:1:99
            
            fls = dir(['bhv_results' num2str(numsim+1) '/' num2str(numiter) '/SUB' sprintf('%03d_BHV.csv',i+1) ]);
            %                         fl = dir(['simul_results/T' num2str(t) '/MODEL7/SUB' sprintf('%03d',i+1) '_SIMUL_BHV.npy']);
            if size(fls,1)==0
                %             batch_simul_gen_by_regression(mod,i,sisi);
                job.name = 'generates.py';
                job.path0 = '/home/sjh/kdj/parameter_recovery/gen_simulation_dat/gail/';
                job.argu = [sprintf('--subid=%d --numsim=%d --numiter=%d',i,numiter,numsim)];
                job.pwd = job.path0;
                job.nth = i;
                JobPython(id,job,'old_torch',1,rem(i,3)+1);
                [out] = SubmitJob(id,pw,job);
            end
        end
    end
end