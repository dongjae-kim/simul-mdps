% batch jobs for 
addpath('\\143.248.32.146\sjh\kdj\TerminalControl');

id = 'sjh';
pw = 'kinggodjh';
%         batch_information_recovery_subjective(i,ktov)
numsim = 2;
for i = 0:1:81
    %                 fl = dir(['simul_results/T' num2str(t) '/MODEL7/SUB' sprintf('%03d',i+1) '_SIMUL_BHV.npy']);
    %                 if size(fl,1)==0
    %             batch_simul_gen_by_regression(mod,i,sisi);
    job.name = 'run.py';
    job.path0 = '/home/sjh/kdj/parameter_recovery/gen_simulation_dat/IQN_PM/';
    job.argu = [sprintf('--subid=%d --numsim=%d',i,numsim)];
    job.pwd = job.path0;
    job.nth = i;
    JobPython(id,job,'Code',1,rem(i,3)+1);
    [out] = SubmitJob(id,pw,job);
end

%% generating simuls
% pause(60*5)
addpath('\\143.248.32.114\sjh\kdj\TerminalControl');

id = 'sjh';
pw = 'kinggodjh';
%         batch_information_recovery_subjective(i,ktov)
numsim = 2;
for i = 0:1:81
    for numiter = 0:1:99
        %                 fl = dir(['simul_results/T' num2str(t) '/MODEL7/SUB' sprintf('%03d',i+1) '_SIMUL_BHV.npy']);
        %                 if size(fl,1)==0
        %             batch_simul_gen_by_regression(mod,i,sisi);
        job.name = 'main_generate.py';
        job.path0 = '/home/sjh/kdj/parameter_recovery/gen_simulation_dat/IQN_PM/';
        job.argu = [sprintf('--subid=%d --numsim=%d --numiter=%d',i,numsim,numiter)];
        job.pwd = job.path0;
        job.nth = i;
        JobPython(id,job,'Code',1,rem(i,3)+1);
        [out] = SubmitJob(id,pw,job);
    end
end