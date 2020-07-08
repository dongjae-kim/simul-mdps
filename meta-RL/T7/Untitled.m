
addpath('\\143.248.32.114\sjh\kdj\TerminalControl');
addpath('\\143.248.32.114\sjh\kdj\TerminalControl');

id = 'sjh';
pw = 'kinggodjh';
%         batch_information_recovery_subjective(i,ktov)
t = 0;
            for i = 0:1:81
                %                 fl = dir(['simul_sresults/T' num2str(t) '/MODEL7/SUB' sprintf('%03d',i+1) '_SIMUL_BHV.npy']);
                %                 if size(fl,1)==0
                %             batch_simul_gen_by_regression(mod,i,sisi);
                job.name = 'main_gen_simul_task_meta_RL_main_force_to_extract_likelihood';
                job.path0 = '/home/sjh/kdj/parameter_recovery/gen_simulation_dat/novel_tasks_python/GPUfit/';
                job.argu = [sprintf(' --sub-id=%d',t,i)];
                job.pwd = job.path0;
                job.nth = i;
                JobPython(id,job,'Code',1,2);
                [out] = SubmitJob(id,pw,job);
            end