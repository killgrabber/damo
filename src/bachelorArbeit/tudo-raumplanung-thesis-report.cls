\ProvidesClass{tudo-raumplanung-thesis-report}[2023/12/02 v1.0 TU Dortmund Spatial Planning Thesis and Report Template]
\NeedsTeXFormat{LaTeX2e}
\LoadClass[titlepage=firstiscover]{scrbook}

% Force XeLaTeX
\RequirePackage{ifxetex}
\RequireXeTeX

\RequirePackage[utf8]{inputenc}
\RequirePackage{graphicx}
\RequirePackage{subfig}
\RequirePackage{url}
\RequirePackage{geometry}
\RequirePackage{xifthen}
\RequirePackage{wrapfig}
\RequirePackage{textpos}
\RequirePackage{babel}
\RequirePackage{multicol}
\RequirePackage{natbib}
\RequirePackage{xcolor}
\RequirePackage[quiet]{fontspec}
\RequirePackage{pdfpages}
\RequirePackage{acronym}
\RequirePackage{forloop}

\defaultfontfeatures{Ligatures=TeX}
% Font Configuration
%\RequirePackage{libertineRoman}
%\RequirePackage{nimbussans}

\setsansfont{NotoSans}[
	  Ligatures={Common,Rare},
		Path=./fonts/NotoSans/,
		Scale=0.9,
		Extension = .ttf,
		UprightFont=*-Regular,
		BoldFont=*-Bold,
		ItalicFont=*-Italic,
		BoldItalicFont=*-BoldItalic]

\setmonofont{Inconsolata}[
				Path=./fonts/Inconsolata/,
				Scale=0.9,
				Extension = .ttf,
				UprightFont=*-Regular,
				BoldFont=*-Bold]

\setmainfont{LinLibertine}%
   [Ligatures={Common,Rare}, Numbers=OldStyle,
	 Path=./fonts/LinLibertine/,
	 Scale=0.9,
	 Extension = .otf,
	 UprightFont=*_R,
	 BoldFont=*_RB,
	 ItalicFont=*_RI,
	 BoldItalicFont=*_RBI]

\RequirePackage[libertine]{newtxmath}

\setkomafont{pagehead}{\sffamily}
\setkomafont{pagenumber}{\sffamily}
\setkomafont{footnote}{\rmfamily}
\setkomafont{caption}{\rmfamily}
\setkomafont{captionlabel}{\rmfamily}

% Listings
\RequirePackage{listings}
\definecolor{KWColor}{rgb}{0.37,0.08,0.25}
\definecolor{CommentColor}{rgb}{0.133,0.545,0.133}
\definecolor{StringColor}{rgb}{0,0.126,0.941}
\lstset{
		language=Java,
        float=th,
        columns=fullflexible,
        basicstyle=\scriptsize\ttfamily,
        numbers=left,
        numberstyle=\tiny\color{black},
        captionpos=b,
        xleftmargin=15pt,
	      tabsize=4,
        numbersep=5pt,
      	stepnumber=1,
        extendedchars=true,
        breaklines=true,
        breakatwhitespace=true,
        showtabs=false,
        showspaces=false,
        showstringspaces=false,
        keywordstyle=\bfseries\color{KWColor},
        commentstyle=\color{CommentColor},
        stringstyle=\color{StringColor},
        escapechar=\%,
 		language=Java,
		escapeinside={(*@}{@*)}
}


% Internationalization
\newcommand{\thesistype}{}
\newcommand{\examination}{}
\newcommand{\appendixtitle}{}
\newcommand{\listabbreviation}{}
\newcommand{\titlelistfigures}{}
\newcommand{\makelistfigures}{}
\newcommand{\makeappendix}{}
\newcommand{\makebibliography}{}

\DeclareOption{german}{%
  \PassOptionsToPackage{main=german,english}{babel}
  \DeclareOption{ba}{\renewcommand{\thesistype}{Bachelorarbeit}}
  \DeclareOption{ma}{\renewcommand{\thesistype}{Masterarbeit}}
  \DeclareOption{expose}{\renewcommand{\thesistype}{Exposé}}
  \DeclareOption{interim}{\renewcommand{\thesistype}{Zwischenbericht}}
  \DeclareOption{report}{\renewcommand{\thesistype}{Projektbericht}}
  \renewcommand{\examination}{Begutachtung:}
  \renewcommand{\appendixtitle}{Anhang}
  \renewcommand{\listabbreviation}{Abkürzungsverzeichnis}
  \renewcommand{\titlelistfigures}{Abbildungsverzeichnis}
  % Bibliography
  \renewcommand*{\makebibliography}{
    \addcontentsline{toc}{chapter}{\bibname}
    \bibliographystyle{apalike-german}
  }
}
\DeclareOption{english}{%
  \PassOptionsToPackage{main=english,german}{babel}
  \DeclareOption{ba}{\renewcommand{\thesistype}{Bachelor Thesis}}
  \DeclareOption{ma}{\renewcommand{\thesistype}{Master Thesis}}
  \DeclareOption{expose}{\renewcommand{\thesistype}{Proposal}}
  \DeclareOption{interim}{\renewcommand{\thesistype}{Interim report}}
  \DeclareOption{report}{\renewcommand{\thesistype}{Project report}}
  \renewcommand{\examination}{Reviewers:}
  \renewcommand{\appendixtitle}{Appendix}
  \renewcommand{\listabbreviation}{List of abbreviations}
  \renewcommand{\titlelistfigures}{List of figures}
  % Bibliography
  \renewcommand*{\makebibliography}{
    \addcontentsline{toc}{chapter}{\bibname}
    \bibliographystyle{apalike}
  }
}

\DeclareOption*{\PackageWarning{tudo-raumplanung-thesis-report}{Unknown ‘\CurrentOption’}}
\ExecuteOptions{english,ba}
\ProcessOptions\relax

% This needs to be added for multilanguage support...
\defcaptionname{english}\abstractname{Abstract}
\defcaptionname{german}\abstractname{Zusammenfassung}

\defcaptionname{english}\disclaimername{Disclaimer on the use of gender-neutral language}
\defcaptionname{german}\disclaimername{Erklärung über den Gebrauch geschlechtsneutraler Sprache}

% Environments for English and German lists of abbreviations
\newenvironment{abbreviations}
{
    \chapter*{\listabbreviation}
    \addcontentsline{toc}{chapter}{\listabbreviation}
    \begin{acronym}[abbreviations]
    \setlength{\parskip}{-1ex}
}
{
    \end{acronym}
    \cleardoublepage
}

% Environments for English and German disclaimers regarding use of gender-neutral language
\newenvironment{disclaimer-gender}
  {%\begin{otherlanguage}{english}
  \chapter*{\disclaimername}
  \addcontentsline{toc}{chapter}{\disclaimername}
  }
  {
  %\end{otherlanguage}
  \cleardoublepage
  }

%\newenvironment{disclaimer-gender-ger}
 % {\begin{otherlanguage}{german}
  %\chapter*{\disclaimername}
  %\addcontentsline{toc}{chapter}{\disclaimername}
  %}
  %{
  %\end{otherlanguage}
  %\cleardoublepage
  %}

% Environments for English and German abstracts
\newenvironment{abstract}
  {\begin{otherlanguage}{english}\chapter*{\abstractname}
  \addcontentsline{toc}{chapter}{\abstractname}
  }
  {
  \end{otherlanguage}
  }

\newenvironment{abstract-ger}
  {\begin{otherlanguage}{german}\chapter*{\abstractname}
  \addcontentsline{toc}{chapter}{\abstractname}
  }
  {
  \end{otherlanguage}
  }

%& Authors
\newcount\authnum
\def\storeauthors#1#2{\authnum=0 \edef\tmp{\string#1}\storeauthorsA#2\end}
\def\storeauthorsA#1{\advance\authnum by1
   \ifx\end#1\else
      \expandafter\def\csname data:\tmp:\the\authnum\endcsname{#1}%
      \expandafter\storeauthorsA\fi
}
\def\getauthor[#1]#2{\csname data:\string#2:#1\endcsname}

%& Examiners/Reviewers
\newcount\reviewersnum
\def\storereviewers#1#2{\reviewersnum=0 \edef\tmp{\string#1}\storereviewersA#2\end}
\def\storereviewersA#1{\advance\reviewersnum by1
   \ifx\end#1\else
      \expandafter\def\csname data:\tmp:\the\reviewersnum\endcsname{#1}%
      \expandafter\storereviewersA\fi
}
\def\getreviewer[#1]#2{\csname data:\string#2:#1\endcsname}

% Title page
\renewcommand*{\makelistfigures}{
  \listoffigures
  \addcontentsline{toc}{chapter}{\titlelistfigures}
}

% Appendix
\renewcommand*{\makeappendix}{
  \addtocontents{toc}{\protect\vspace*{\baselineskip}}
  \appendix
  \renewcommand\chaptername{\appendixtitle}
}

\renewcommand*{\coverpagetopmargin}{100mm}
\renewcommand*{\coverpageleftmargin}{100mm}
\renewcommand*{\coverpagerightmargin}{0mm}
\renewcommand*{\coverpagebottommargin}{0mm}

\renewcommand*{\maketitle}[1][1]{

  \newgeometry{top=4cm,outer=3cm,inner=3cm,bottom=5cm}
  \begin{titlepage}
    \cleardoublepage
    \pagestyle{empty}

    \begin{textblock}{8}(3.1,11)
      \small \raggedright \noindent \sffamily
      \setlength{\columnsep}{-6.9cm}
      \begin{multicols}{2}
        \includegraphics[width=1.5cm]{logos/rp_logo.pdf}
        \columnbreak

        Technische Universität Dortmund \\
        Fakultät Raumplanung
      \end{multicols}
    \end{textblock}

    \centering\sffamily

    \includegraphics[width=9cm]{logos/tud_logo_cmyk.pdf}

    \vspace{0.2cm}

    {\thesistype}

    \vspace{1cm}

    {\Huge\bfseries\@title}

    \vspace{1cm}

    \ifnum\authnum>11
      \begin{multicols}{2}
    \fi

    % Print authors' names
    \ifnum\authnum>2 %
      \newcounter{x}
      \forloop{x}{1}{\value{x} < \authnum}{
        \ifnum\value{x}<2
          {\Large\getauthor[\thex]\authors\\}
        \else
          \vspace{0.1cm}
          {\Large\getauthor[\thex]\authors\\}
        \fi
      }      
    \else
      {\Large\getauthor[1]\authors}
    \fi

    \ifnum\authnum>11
      \end{multicols}
    \fi

    \addvspace{0.5cm}
    \vfill

    {\large\@date}

    %\vspace{2cm}

    \vfill

    \examination \\
    % Print reviewers' names
    \ifnum\reviewersnum>2 %
      \newcounter{y}
      \forloop{y}{1}{\value{y} < \reviewersnum}{
        {\getreviewer[\they]\reviewers\\}
      }      
    \else
      {\getreviewer[1]\reviewers}
    \fi

    \vfill

    \pagebreak


    \cleardoublepage

  \end{titlepage}
  \restoregeometry
}
