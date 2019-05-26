#!/bin/perl
#evaluate the dp results for 08-form
#usage: perl ... gold test
use strict;
use warnings;
my $all_sentences;
my $all_sentences_correct;
my $all_sentences_nopunc_correct;
my $all_tokens;
my $all_tokens_correct;
my $all_tokens_nopunc;
my $all_tokens_nopunc_correct;
my $all_root;
my $all_root_correct;
open FGOLD,"<$ARGV[0]";
open FTEST,"<$ARGV[1]";
my $curr_sent_right = 1;
my $curr_sent_right_nonpunc = 1;
my $curr_sent_count = 0;
# for PTB, UD, CTB
my %punc_set = (",",1,".",1,"``",1,"''",1,":",1,"PUNCT",1,"SYM",1,"PU",1);
while(<FGOLD>){
    my $line_t = <FTEST>;
    if(!/[0-9]/){
        if($curr_sent_count > 0){
            #finish one sentences
            $all_sentences ++;
            $all_sentences_correct ++ if $curr_sent_right != 0;
            $all_sentences_nopunc_correct ++ if $curr_sent_right_nonpunc != 0;
        }
        $curr_sent_right = 1;
        $curr_sent_right_nonpunc = 1;
        $curr_sent_count = 0;
    }
    else{
        my @lg = split;
        my @lt = split /\s/,$line_t;
        $all_tokens ++;
        $curr_sent_count ++;
        $all_root ++ if ($lg[6]+0) == 0;
        $all_tokens_nopunc ++ if(! $punc_set{$lg[3]});
        die "Not correct ($lg[0] ne $lt[0]) or ($lg[1] ne $lt[1])" if ($lg[0] ne $lt[0]) or (($lg[1] ne $lt[1]) and ($lt[1] ne "<num>"));
        if ($lg[6] eq $lt[6]){
            $all_tokens_correct++;
            $all_tokens_nopunc_correct++ if(! $punc_set{$lg[3]});
            $all_root_correct++ if ($lg[6]+0) == 0;
        }
        else{
            $curr_sent_right = 0;
            $curr_sent_right_nonpunc = 0 if(! $punc_set{$lg[3]});
        }
    }
}
printf "Tokens:%d/%d/%g\n",$all_tokens,$all_tokens_correct,$all_tokens_correct/$all_tokens;
printf "Tokens-nonpunc:%d/%d/%g\n",$all_tokens_nopunc,$all_tokens_nopunc_correct,$all_tokens_nopunc_correct/$all_tokens_nopunc;
printf "Sentence:%d/%d/%g\n",$all_sentences,$all_sentences_correct,$all_sentences_correct/$all_sentences;
printf "Sentence-nonpunc:%d/%d/%g\n",$all_sentences,$all_sentences_nopunc_correct,$all_sentences_nopunc_correct/$all_sentences;
printf "Root:%d/%d/%g\n",$all_root,$all_root_correct,$all_root_correct/$all_root;

