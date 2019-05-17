% from a daily time series calculate the break-up start and end dates

close all
clear all

% programname='CalculateFUBUDatesBarrowCoastal.m';
% disp(programname)

%load BarrowCoastalDailyTimeseries13 %dailyvalues dailymdates 
load BarrowCoastalDailyTimeseries11
% load BRWCriticalCryoDataFUBUDates %BarrowObservedFreezeUpDates BarrowObservedBreakUpDates  THESE ARE THE OBSERVED DATES
load BarrowCoastalDailySICvalues2

% plot freeze up observed and calculated

% setupfigure
% trimjet2
% GetGreys

% subplot(2,1,1)
% hold on

% calcuatel Freeze Up 

% make "summer" (August and September) means

[allyears,allmonths,~,~,~]=datevec(dailymdates);
eachyear=unique(allyears);
tyears=length(eachyear);

winters=NaN*zeros(tyears,1);
winterdates=NaN*zeros(tyears,1);
FreezeUpEnd=NaN*zeros(tyears,1);
FreezeUpEndConc=NaN*zeros(tyears,1);

summers=NaN*zeros(tyears,1);
summerstds=NaN*zeros(tyears,1);
summerdates=NaN*zeros(tyears,1);
FreezeUpStart=NaN*zeros(tyears,1);
FreezeUpStartConc=NaN*zeros(tyears,1);

% xcorr=[];
% ycorr=[];

%find the average concentration for January through February
for t=1:tyears-1
    thisset=find(allyears==eachyear(t+1) & (allmonths==1 | allmonths==2));
    tmp=dailyvalues(thisset);
    goodones=find(isfinite(tmp));
    % winters(t)=mean(tmp(goodones));
    winters(t+1)=mean(tmp(goodones)); %[MLCHANGED]
    winterdates(t)=mean(dailymdates(thisset));
end

% find average concentration for August through Sept
for t=2:tyears
    thisset=find(allyears==eachyear(t) & (allmonths==8 | allmonths==9));
    tmp=dailyvalues(thisset);
    goodones=find(isfinite(tmp));
    summers(t)=mean(tmp(goodones));
    summerstds(t)=std(tmp(goodones));
    summerdates(t)=mean(dailymdates(thisset));
end

% find the first data where sea-ice conc exceeds summer value plus 1std
disp(' ')
disp('Barrow Freeze Up Start');
disp('      c a l c u l a t e d       o b s e r v e d ');
disp(' year    mon     day     jday   year    mon     day     jday   ');

for t=2:tyears-1  %1978 and last year don't have enough Aug/Sept values
    startday=find(dailymdates==datenum(eachyear(t),9,1));
    thresholds=max(summers(t)+summerstds(t),15);
    while dailyvalues(startday)<=thresholds
          startday=startday+1;
    end
    FreezeUpStart(t)=dailymdates(startday);
    FreezeUpStartConc(t)=dailyvalues(startday);
    
    % [cyear,cmonth,cday]=datevec(FreezeUpStart(t));
    % jcday=FreezeUpStart(t)-datenum(cyear,1,0);
    % [oyear,omonth,oday]=datevec(BarrowObservedFreezeUpDates);
    % matchyear=find(oyear==cyear);
    % joday=BarrowObservedFreezeUpDates(matchyear)-datenum(oyear(matchyear),1,0);
    % if isfinite(matchyear)
    %    junk=[cyear,cmonth,cday,jcday,oyear(matchyear),omonth(matchyear),oday(matchyear),joday];
    %    fprintf(1,'%5.0f\t %3.0f\t %3.0f\t %4.0f\t %3.0f\t %3.0f\t %3.0f\t %3.0f\n',junk);
    % else
    %    junk=[cyear,cmonth,cday,jcday,];
    %    fprintf(1,'%5.0f\t %3.0f\t %3.0f\t %4.0f\n',junk);
    % end

    % if isfinite(joday) & isfinite(jcday)
    %    plot(joday,jcday,'r*');
    %    text(joday,jcday,num2str(cyear),...
    %         'Color',darker,...
    %         'HorizontalAlignment','center',... 
    %         'VerticalAlignment','bottom',... 
    %         'FontSize',9,'FontName','Helvetica',...
    %         'Units','data');
    %     xcorr=[xcorr joday];
    %     ycorr=[ycorr jcday];
    % end
end
% [r,~]=corrcoef(xcorr,ycorr);

% axis square
% set(gca,'xlim',[200 365]);
% set(gca,'ylim',[200 365]);
% line([200 365],[200 365],'Color','k');
% xlabel('observed')
% ylabel('calculated')
% title(['Freeze Up Start Coastal 11. R: ' num2str(r(1,2))])


% subplot(2,1,2)
% hold on

% clear xcorr ycorr
% xcorr=[];
% ycorr=[];

% find the first data where sea-ice conc exceeds winter value minus 10%
disp(' ')
disp('Freeze Up End');
disp(' year    mon     day     jday    conc  threshold  duration');
for t=2:tyears-1  %1978 and last year don't have enough Aug/Sept values
    startday=find(dailymdates==FreezeUpStart(t)+1); %start search 1 day after freeze-up Starts
    if isempty(startday);startday=find(dailymdates==datenum(eachyear(t),9,1));end
    threshold=winters(t)-10;
    while dailyvalues(startday)<=threshold
          startday=startday+1;
    end
    FreezeUpEnd(t)=dailymdates(startday);
    FreezeUpEndConc(t)=dailyvalues(startday);
  
    % [cyear,cmonth,cday]=datevec(FreezeUpEnd(t));
    % jcday=FreezeUpEnd(t)-datenum(cyear,1,0);
    % [oyear,omonth,oday]=datevec(BarrowObservedFreezeUpDates);
    % matchyear=find(oyear==cyear);
    % joday=BarrowObservedFreezeUpDates(matchyear)-datenum(oyear(matchyear),1,0);
    % if isfinite(matchyear)
    %    junk=[cyear,cmonth,cday,jcday,oyear(matchyear),omonth(matchyear),oday(matchyear),joday];
    %    fprintf(1,'%5.0f\t %3.0f\t %3.0f\t %4.0f\t %3.0f\t %3.0f\t %3.0f\t %3.0f\n',junk);
    % else
    %    junk=[cyear,cmonth,cday,jcday,];
    %    fprintf(1,'%5.0f\t %3.0f\t %3.0f\t %4.0f\n',junk);
    % end
    

    % if isfinite(joday) & isfinite(jcday)
    %    plot(joday,jcday,'r*');
    %    text(joday,jcday,num2str(cyear),...
    %         'Color',darker,...
    %         'HorizontalAlignment','center',... 
    %         'VerticalAlignment','bottom',... 
    %         'FontSize',9,'FontName','Helvetica',...
    %         'Units','data');
    %     xcorr=[xcorr joday];
    %     ycorr=[ycorr jcday];
    % end
        
end
% [r,~]=corrcoef(xcorr,ycorr);

% axis square
% set(gca,'xlim',[200 365]);
% set(gca,'ylim',[200 365]);
% line([200 365],[200 365],'Color','k');
% xlabel('observed')
% ylabel('calculated')
% title(['Freeze Up End Coastal 11. R: ' num2str(r(1,2))])
% footer

% BarrowFreezeUpStart=FreezeUpStart(2:tyears-1);
% save BarrowCoastalCalculatedFreezeUpStart11 FreezeUpStart FreezeUpStartConc eachyear

% BarrowFreezeUpEnd=FreezeUpEnd(2:tyears-1);
% save BarrowCoastalCalculatedFreezeUpEnd11 FreezeUpEnd FreezeUpEndConc eachyear

% calculation for break up

winters=NaN*zeros(tyears,1);
winterstd=NaN*zeros(tyears,1);
winterdates=NaN*zeros(tyears,1);
BreakUpStart=NaN*zeros(tyears,1);
BreakUpStartConc=NaN*zeros(tyears,1);

summers=NaN*zeros(tyears,1);
summerstd=NaN*zeros(tyears,1);
summerdates=NaN*zeros(tyears,1);
BreakUpEnd=NaN*zeros(tyears,1);
BreakUpEndConc=NaN*zeros(tyears,1);

% do the plotting  <----------------------------------------------------------------------------------------plotting starts here
% setupfiguresquare

% subplot(1,1,1)
% hold on

% xcorr=[];
% ycorr=[];
%find average January - February concentration, start with 1979
for t=2:tyears-1
    thisset=find(allyears==eachyear(t) & (allmonths==1 | allmonths==2));
    tmp=dailyvalues(thisset);
    goodones=find(isfinite(tmp));
    winters(t)=mean(tmp(goodones));
    winterstd(t)=std(tmp(goodones));
    winterdates(t)=mean(tmp(goodones));
end

% find average concentration for August through Sept
for t=2:tyears-1
    thisset=find(allyears==eachyear(t) & (allmonths==8 | allmonths==9));
    tmp=dailyvalues(thisset);
    goodones=find(isfinite(tmp));
    summers(t)=mean(tmp(goodones));
    summerstd(t)=std(tmp(goodones));
    summerdates(t)=mean(tmp(goodones));
end

% "last day for which previous two weeks are below threshold -2std"
disp(' ')
disp('Barrow Break Up');
disp('      c a l c u l a t e d       o b s e r v e d ');
disp(' year    mon     day     jday   year    mon     day     jday   ');
for t=2:tyears-1 
    threshold=winters(t)-2*winterstd(t);
    %threshold=winters(t)-1.5*winterstd(t);
    %threshold=winters(t)-1*winterstd(t);
    %threshold=winters(t)-0.5*winterstd(t);
    % start search on Feb14 and go through August 1.
    startday=find(dailymdates==datenum(eachyear(t),2,14)); 
    while dailymdates(startday)<=datenum(eachyear(t),8,1);
          allabove=1;
          for twoweeks=0:13
              thisday=startday-twoweeks;
              if dailyvalues(thisday)<threshold;
                 allabove=0;
              end
          end
          if allabove;
             BreakUpStart(t)=dailymdates(startday);
             BreakUpStartConc(t)=dailyvalues(startday);
          end
          startday=startday+1;
    end
    if summers(t)>40;
       BreakUpStart(t)=NaN;
       BreakUpStartConc(t)=NaN;
    end
    if BreakUpStart(t)==datenum(eachyear(t),8,1)
       BreakUpStart(t)=NaN;
       BreakUpStartConc(t)=NaN;
    end

    % [cyear,cmonth,cday]=datevec(BreakUpStart(t));
    % jcday=BreakUpStart(t)-datenum(cyear,1,0);
    % [oyear,omonth,oday]=datevec(BarrowObservedBreakUpDates);
    % matchyear=find(oyear==cyear);
    % joday=BarrowObservedBreakUpDates(matchyear)-datenum(oyear(matchyear),1,0);
    % if isfinite(matchyear)
    %    junk=[cyear,cmonth,cday,jcday,oyear(matchyear),omonth(matchyear),oday(matchyear),joday];
    %    fprintf(1,'%5.0f\t %3.0f\t %3.0f\t %4.0f\t %3.0f\t %3.0f\t %3.0f\t %3.0f\n',junk);
    % else
    %    junk=[cyear,cmonth,cday,jcday,];
    %    fprintf(1,'%5.0f\t %3.0f\t %3.0f\t %4.0f\n',junk);
    % end
    % if isfinite(joday) & isfinite(jcday)
    %    plot(joday,jcday,'r*');
    %    text(joday,jcday,num2str(cyear),...
    %         'Color',darker,...
    %         'HorizontalAlignment','center',... 
    %         'VerticalAlignment','bottom',... 
    %         'FontSize',9,'FontName','Helvetica',...
    %         'Units','data');
    %     xcorr=[xcorr joday];
    %     ycorr=[ycorr jcday];
    % end
end

% [r,p]=corrcoef(xcorr,ycorr);
% set(gca,'xlim',[100 200]);
% set(gca,'ylim',[100 200]);
% line([100 200],[100 200],'Color','k');
% xlabel('observed')
% set(get(gca,'YLabel'),'String','');bel('calculated')
% title(['Break Up Coastal 15. R: ' num2str(r(1,2))])
% footer


% find the last day where conc exceeds summer value plus 1std
disp(' ')
disp('Break Up End Barrow');
disp(' year     year    mon     day    meanconc @breakup threshold duration');
for t=2:tyears-1  %1978 and last year don't have enough Aug/Sept values
    threshold=summers(t)+summerstd(t);
    if threshold<15;threshold=15;end;
    startday=find(dailymdates==datenum(eachyear(t),6,1)); % start June 1
    while dailymdates(startday)<=datenum(eachyear(t),9,31)  %end Sept 31 
          if dailyvalues(startday)>threshold
             BreakUpEnd(t)=dailymdates(startday);
             BreakUpEndConc(t)=dailyvalues(startday);
          end
          startday=startday+1;
    end
    if summers(t)>25;
       BreakUpEnd(t)=NaN;
       BreakUpEndConc(t)=NaN;
    end
    if BreakUpEnd(t)==datenum(eachyear(t),9,31);
       BreakUpEnd(t)=NaN;
       BreakUpEndConc(t)=NaN;
    end
    if isfinite(BreakUpEnd(t)) && isfinite(BreakUpStart(t))
       duration=BreakUpEnd(t)-BreakUpStart(t);
    else
       duration=NaN;
    end
    [iyear,imonth,iday]=datevec(BreakUpEnd(t));
    jday=BreakUpEnd(t)-datenum(eachyear(t),1,1)+1;
    junk=[eachyear(t),iyear,imonth,iday,jday,BreakUpEndConc(t),threshold,duration];
    fprintf(1,'%5.0f\t %5.0f\t %3.0f\t %3.0f\t %4.0f\t %5.2f\t %5.2f\t %5.0f \n',junk);
end


BarrowBreakUpStart=BreakUpStart(2:tyears-1);
save BarrowCoastalCalculatedBreakUpStart11 BreakUpStart BreakUpStartConc eachyear

BarrowBreakUpEnd=BreakUpEnd(2:tyears-1);
save BarrowCoastalCalculatedBreakUpEnd11 BreakUpEnd BreakUpEndConc eachyear


% do the plotting  <-----------------------------------------------plotting starts here
down_scale=0.75;
setupfigure

subplot(1,1,1)
hold on
plot(dailymdates,dailyvalues,'b-','Color',light,'LineWidth',1);

plot(FreezeUpStart,FreezeUpStartConc,'bs','MarkerSize',15,'LineWidth',2);
plot(FreezeUpEnd,FreezeUpEndConc,'bs','MarkerSize',11,'LineWidth',2);

for t=2:tyears-1
    line([FreezeUpStart(t) FreezeUpStart(t)],[0 102],'Color','b','LineStyle',':');
    line([FreezeUpEnd(t) FreezeUpEnd(t)],[-3 100],'Color','b','LineStyle',':','Clipping','off');

    if isnan(FreezeUpStart(t));
       str1='NaN';
    else
       str1=datestr(FreezeUpStart(t),'mm/dd');
    end
    text(FreezeUpStart(t),103,str1,...
        'HorizontalAlignment','center','FontSize',7,'FontName','Arial','Color','b','Units','data');
    
    if isnan(FreezeUpEnd(t));
       str1='NaN';
    else
       str1=datestr(FreezeUpEnd(t),'mm/dd');
    end
    text(FreezeUpEnd(t),-4,str1,...  %was 101
        'HorizontalAlignment','center','FontSize',7,'FontName','Arial','Color','b','Units','data');     
end

for n=1:length(BarrowObservedFreezeUpDates)
    day=BarrowObservedFreezeUpDates(n);
    line([BarrowObservedFreezeUpDates(n) BarrowObservedFreezeUpDates(n)],[0 100],'Color','b','LineStyle','-','LineWidth',2);
end


plot(BreakUpStart,BreakUpStartConc,'rs','MarkerSize',15,'LineWidth',2);
plot(BreakUpEnd,BreakUpEndConc,'rs','MarkerSize',11,'LineWidth',2);

for t=2:tyears-1
    line([BreakUpStart(t) BreakUpStart(t)],[0 102],'Color','r','LineStyle',':');
    %line([winterdates(t)-30 winterdates(t)+30],[winters(t) winters(t)],'Color',light,'LineStyle','-','LineWidth',6);    
    %line([winterdates(t) BreakUpStart(t)-14],[winters(t)-2*winterstd(t) winters(t)-2*winterstd(t)],'Color',light,'LineStyle','-','LineWidth',4);
    
    line([BreakUpEnd(t) BreakUpEnd(t)],[-3 100],'Color','r','LineStyle',':','Clipping','off');
    %line([summerdates(t)-30 summerdates(t)+30],[summers(t) summers(t)],'Color',light,'LineStyle','-','LineWidth',6);
    %line([BreakUpEnd(t) summerdates(t)],[summers(t)+summerstd(t) summers(t)+summerstd(t)],'Color',light,'LineStyle','-','LineWidth',4) 

    if isnan(BreakUpStart(t));
       str1='NaN';
    else
       str1=datestr(BreakUpStart(t),'mm/dd');
    end
    text(BreakUpStart(t),103,str1,...
        'HorizontalAlignment','center','FontSize',7,'FontName','Arial','Color','r','Units','data');
    
    if isnan(BreakUpEnd(t));
       str1='NaN';
    else
       str1=datestr(BreakUpEnd(t),'mm/dd');
    end
    text(BreakUpEnd(t),-4,str1,...  %was 101
        'HorizontalAlignment','center','FontSize',7,'FontName','Arial','Color','r','Units','data');     
end

for n=1:length(BarrowObservedBreakUpDates)
    day=BarrowObservedBreakUpDates(n);
    line([BarrowObservedBreakUpDates(n) BarrowObservedBreakUpDates(n)],[0 100],'Color','r','LineStyle','-','LineWidth',2);
end




%for the entire time series
mstart=datenum(1978,11,1);
mend=datenum(2014,1,1);
%for some close up
%mstart=datenum(1980,6,1,0,0,0);
%mend=datenum(2010,2,1,0,0,0);
%mstart=datenum(1991,6,1,0,0,0);
%mend=datenum(1996,2,1,0,0,0);
%mstart=datenum(1996,6,1,0,0,0);
%mend=datenum(2001,2,1,0,0,0);
%mstart=datenum(2001,6,1,0,0,0);
%mend=datenum(2005,2,1,0,0,0);
%mstart=datenum(2005,6,1,0,0,0);
%mend=datenum(2009,2,1,0,0,0);

set(gca,'Xlim',[mstart mend]);
set(gca,'XTick',datenum(1978:1:2014,1,1));
datetick('x','mmmyy','keeplimits','keepticks')
set(gca,'Ylim',[0 105]);
ylabel('sea ice concentration in %')

footer
hold on
text(.5,1.03,'Barrow FreezeUp and Break Up Dates',...
      'Color',darker,...
      'HorizontalAlignment','center',... 
      'VerticalAlignment','bottom',... 
      'FontSize',14,'FontName','Helvetica',...
      'Units','normalized');
text(.48,-0.075,'circles: observed',...
      'Color',darker,...
      'HorizontalAlignment','right',... 
      'VerticalAlignment','bottom',... 
      'FontSize',9,'FontName','Helvetica',...
      'Units','normalized');
text(.52,-0.075,'squares: calculated',...
      'Color',darker,...
      'HorizontalAlignment','left',... 
      'VerticalAlignment','bottom',... 
      'FontSize',9,'FontName','Helvetica',...
      'Units','normalized');