/**
 * PDF Generator for Logistic Regression Experiment
 * High-fidelity notebook-style PDF generation using jsPDF and jsPDF-AutoTable
 */

async function downloadPDF() {
    if (typeof window.jspdf === 'undefined') {
        alert('PDF library not loaded. Please try again.');
        return;
    }

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
    });

    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 15;
    const contentWidth = pageWidth - (margin * 2);
    let yPos = margin;

    const FONT = 'helvetica';
    doc.setFont(FONT, 'normal');

    function checkNewPage(requiredHeight) {
        if (yPos + requiredHeight > pageHeight - margin) {
            doc.addPage();
            yPos = margin;
            doc.setFont(FONT, 'normal');
            return true;
        }
        return false;
    }

    function decodeAndStripHTML(html) {
        if (!html) return '';
        const txt = document.createElement('textarea');
        txt.innerHTML = html.replace(/<br\s*\/?>/gi, '\n').replace(/&nbsp;/g, ' ');
        const decoded = txt.value;
        return decoded.replace(/<[^>]*>/g, '').trim();
    }

    async function addImageFromUrl(url, widthPercent = 0.9) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = "Anonymous";
            img.onload = () => {
                const aspectRatio = img.height / img.width;
                let imgWidth = contentWidth * widthPercent;
                let imgHeight = imgWidth * aspectRatio;

                if (imgHeight > pageHeight - (margin * 2) - 10) {
                    imgHeight = pageHeight - (margin * 2) - 20;
                    imgWidth = imgHeight / aspectRatio;
                }

                checkNewPage(imgHeight + 5);
                doc.addImage(img, 'PNG', margin + (contentWidth - imgWidth) / 2, yPos, imgWidth, imgHeight);
                yPos += imgHeight + 8;
                resolve();
            };
            img.onerror = () => { resolve(); };
            img.src = url;
        });
    }

    async function renderElementRecursively(el) {
        if (el.nodeType === Node.TEXT_NODE) {
            const text = el.textContent.trim();
            if (text) {
                doc.setFontSize(9.5);
                const lines = doc.splitTextToSize(text, contentWidth - 10);
                const h = lines.length * 5 + 4;
                checkNewPage(h);
                doc.setTextColor(40, 40, 40);
                doc.text(lines, margin + 5, yPos + 5);
                yPos += h + 2;
            }
            return;
        }

        if (el.nodeType !== Node.ELEMENT_NODE) return;

        // Table Handling
        if (el.tagName === 'TABLE') {
            let tableEl = el;
            if (el.id) {
                const live = document.getElementById(el.id);
                if (live && live.rows.length > 1) tableEl = live;
            }

            doc.autoTable({
                html: tableEl,
                startY: yPos,
                margin: { left: margin, right: margin },
                styles: { fontSize: 8, font: FONT, cellPadding: 2, overflow: 'linebreak' },
                headStyles: { fillColor: [240, 240, 240], textColor: [40, 40, 40], fontStyle: 'bold' },
                theme: 'grid'
            });
            yPos = doc.lastAutoTable.finalY + 8;
            return;
        }

        // Image Handling
        if (el.tagName === 'IMG') {
            const src = el.getAttribute('src').replace(/^\.\//, '');
            await addImageFromUrl(src);
            return;
        }

        // Specific Container Handling (Smart Live Capture)
        let dataSource = el;
        if (el.id) {
            const live = document.getElementById(el.id);
            if (live && live.innerHTML.trim() !== "" && live.style.display !== 'none') dataSource = live;
        }

        // If it's a simple text-carrying element like P, B, I, or a small div
        if (['P', 'B', 'I', 'SPAN', 'STRONG'].includes(el.tagName) || (el.tagName === 'DIV' && !el.querySelector('table, img, div'))) {
            const txt = decodeAndStripHTML(dataSource.innerHTML).trim();
            if (txt) {
                doc.setFontSize(el.tagName === 'B' || el.tagName === 'STRONG' ? 10 : 9.5);
                doc.setFont(FONT, (el.tagName === 'B' || el.tagName === 'STRONG') ? 'bold' : 'normal');
                const lines = doc.splitTextToSize(txt, contentWidth - 10);
                const h = lines.length * 5 + 6;
                checkNewPage(h);

                if (el.classList.contains('output-success') || el.classList.contains('output-header')) {
                    doc.setFillColor(240, 248, 255);
                    doc.setDrawColor(120, 173, 28);
                    doc.roundedRect(margin, yPos, contentWidth, h, 1, 1, 'FD');
                    doc.setTextColor(29, 53, 87);
                    doc.text(lines, margin + 5, yPos + 6);
                    yPos += h + 6;
                } else {
                    doc.setTextColor(40, 40, 40);
                    doc.text(lines, margin + 5, yPos + 6);
                    yPos += h + 2;
                }
                doc.setFont(FONT, 'normal');
            }
        } else {
            // Recurse into children
            const children = Array.from(dataSource.childNodes);
            for (const child of children) {
                await renderElementRecursively(child);
            }
        }
    }

    // Cover Page
    doc.setFontSize(22);
    doc.setFont(FONT, 'bold');
    doc.setTextColor(29, 53, 87);
    doc.text('Logistic Regression Experiment Report', pageWidth / 2, yPos, { align: 'center' });
    yPos += 12;

    doc.setFontSize(11);
    doc.setFont(FONT, 'normal');
    doc.setTextColor(100, 100, 100);
    doc.text('Virtual Labs - IIT Delhi | Laboratory Notebook', pageWidth / 2, yPos, { align: 'center' });
    yPos += 8;
    doc.setDrawColor(200, 200, 200);
    doc.line(margin, yPos, pageWidth - margin, yPos);
    yPos += 15;

    for (let sIdx = 0; sIdx < stepsData.length; sIdx++) {
        const step = stepsData[sIdx];
        checkNewPage(30);

        doc.setFontSize(16);
        doc.setFont(FONT, 'bold');
        doc.setTextColor(42, 157, 143);
        doc.text(`Step ${sIdx + 1}: ${step.title}`, margin, yPos);
        yPos += 10;

        for (const block of step.blocks) {
            checkNewPage(20);

            const commentMatch = block.code.match(/#\s*([^<\n\r]*)/);
            if (commentMatch) {
                doc.setFontSize(11);
                doc.setFont(FONT, 'italic');
                doc.setTextColor(100, 100, 100);
                const commentTxt = decodeAndStripHTML(commentMatch[1]);
                const lines = doc.splitTextToSize(`# ${commentTxt}`, contentWidth);
                doc.text(lines, margin, yPos);
                yPos += lines.length * 5 + 4;
            }

            const pureCode = decodeAndStripHTML(block.code.replace(/#\s*[^<\n\r]*/, '')).trim();
            if (pureCode) {
                doc.setFontSize(9.5);
                doc.setFont(FONT, 'normal');
                const codeLines = doc.splitTextToSize(pureCode, contentWidth - 12);
                const h = codeLines.length * 5.2 + 8;
                checkNewPage(h + 5);
                doc.setFillColor(248, 250, 252);
                doc.setDrawColor(89, 148, 211);
                doc.roundedRect(margin, yPos, contentWidth, h, 1.5, 1.5, 'FD');
                doc.setTextColor(40, 40, 40);
                doc.text(codeLines, margin + 6, yPos + 7);
                yPos += h + 8;
            }

            if (block.output) {
                const parser = new DOMParser();
                const html = parser.parseFromString(block.output, 'text/html');
                await renderElementRecursively(html.body);
            }
            yPos += 4;
        }
        yPos += 8;
    }

    const totalPages = doc.internal.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
        doc.setPage(i);
        doc.setFontSize(9);
        doc.setTextColor(150, 150, 150);
        doc.text(`Page ${i} of ${totalPages}`, pageWidth / 2, pageHeight - 10, { align: 'center' });
        doc.text('© Virtual Labs IIT Delhi', margin, pageHeight - 10);
    }

    doc.save('Logistic_Regression_Experiment_Notebook.pdf');
}
