// PDF Download Logic
async function downloadPDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    let yPos = 20;

    // Title
    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.text("Logistic Regression - Complete Training", 20, yPos);
    yPos += 10;

    // Red line under title
    doc.setDrawColor(239, 83, 80);
    doc.setLineWidth(0.5);
    doc.line(20, yPos, 190, yPos);
    yPos += 12;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);

    for (const [stepIndex, step] of stepsData.entries()) {
        // Page break check
        if (yPos > 240) {
            doc.addPage();
            yPos = 20;
        }

        // Step Title with red left bar
        doc.setDrawColor(239, 83, 80);
        doc.setFillColor(239, 83, 80);
        doc.rect(20, yPos - 3, 2, 8, 'F');

        doc.setFont("helvetica", "bold");
        doc.setFontSize(12);
        doc.setTextColor(0, 0, 0);
        doc.text(`${stepIndex + 1}. ${step.title}`, 25, yPos + 3);
        yPos += 12;

        // Process each code block
        for (const [blockIndex, block] of step.blocks.entries()) {
            // Page break check
            if (yPos > 235) {
                doc.addPage();
                yPos = 20;
            }

            // Code Block Label
            doc.setFont("helvetica", "bold");
            doc.setFontSize(9);
            doc.setTextColor(100, 100, 100);
            const blockLabel = step.blocks.length > 1 ? `Code Block ${blockIndex + 1}:` : "Code Block 1:";
            doc.text(blockLabel, 25, yPos);
            yPos += 6;

            // Clean code
            let cleanCode = block.code
                .replace(/<div.*?>|<\/div>/gi, '\n')
                .replace(/<.*?>/g, '')
                .trim();

            // Limit code lines
            const codeLines = cleanCode.split('\n');
            const maxCodeLines = 15;
            const limitedCodeLines = codeLines.slice(0, maxCodeLines);
            if (codeLines.length > maxCodeLines) {
                limitedCodeLines.push('... (code continues)');
            }
            const limitedCode = limitedCodeLines.join('\n');

            const splitCode = doc.splitTextToSize(limitedCode, 160);

            // Check fit
            if (yPos + (splitCode.length * 4) + 15 > 275) {
                doc.addPage();
                yPos = 20;
            }

            // Draw gray background for code
            const codeHeight = Math.min((splitCode.length * 4) + 6, 70);
            doc.setFillColor(240, 240, 240);
            doc.rect(25, yPos - 3, 160, codeHeight, 'F');

            // Code text
            doc.setFont("courier", "normal");
            doc.setFontSize(7);
            doc.setTextColor(50, 50, 50);
            doc.text(splitCode, 27, yPos + 2);
            yPos += codeHeight + 6;

            // Output Label
            doc.setFont("helvetica", "bold");
            doc.setFontSize(9);
            doc.setTextColor(100, 100, 100);
            doc.text("Output:", 25, yPos);
            yPos += 6;

            // Parse output
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = block.output;

            // Check for tables
            const tables = tempDiv.querySelectorAll('table');

            if (tables.length > 0) {
                // Handle tables
                for (const table of tables) {
                    const headers = [];
                    const headerCells = table.querySelectorAll('thead th');
                    headerCells.forEach(cell => headers.push(cell.textContent.trim()));

                    const rows = [];
                    const bodyRows = table.querySelectorAll('tbody tr');
                    bodyRows.forEach(row => {
                        const rowData = [];
                        const cells = row.querySelectorAll('td');
                        cells.forEach(cell => rowData.push(cell.textContent.trim()));
                        rows.push(rowData);
                    });

                    const maxRows = 10;
                    const limitedRows = rows.slice(0, maxRows);
                    if (rows.length > maxRows) {
                        limitedRows.push(headers.map(() => '...'));
                    }

                    if (yPos > 240) {
                        doc.addPage();
                        yPos = 20;
                    }

                    doc.autoTable({
                        head: [headers],
                        body: limitedRows,
                        startY: yPos,
                        margin: { left: 25, right: 25 },
                        styles: { fontSize: 7, cellPadding: 2 },
                        headStyles: { fillColor: [240, 240, 240], textColor: [50, 50, 50] },
                        theme: 'grid',
                        tableWidth: 160
                    });

                    yPos = doc.lastAutoTable.finalY + 8;
                }
            }

            // Check for images
            const images = tempDiv.querySelectorAll('img');

            if (images.length > 0) {
                for (const img of images) {
                    // Get image source
                    const imgSrc = img.getAttribute('src');

                    if (imgSrc) {
                        try {
                            // Load image and embed it
                            await new Promise((resolve, reject) => {
                                const image = new Image();
                                image.crossOrigin = "Anonymous";
                                image.onload = function () {
                                    // Check page space
                                    const imgHeight = 80; // Fixed height for consistency
                                    if (yPos + imgHeight + 10 > 275) {
                                        doc.addPage();
                                        yPos = 20;
                                    }

                                    // Add image to PDF
                                    try {
                                        doc.addImage(image, 'PNG', 25, yPos, 140, imgHeight);
                                        yPos += imgHeight + 8;
                                    } catch (e) {
                                        console.error('Error adding image:', e);
                                    }
                                    resolve();
                                };
                                image.onerror = function () {
                                    console.error('Failed to load image:', imgSrc);
                                    resolve(); // Continue even if image fails
                                };
                                image.src = imgSrc;
                            });
                        } catch (e) {
                            console.error('Error processing image:', e);
                        }
                    }
                }
            }

            // Handle text output
            let cleanOutput = tempDiv.textContent
                .replace(/\s+/g, ' ')
                .trim();

            if (cleanOutput && cleanOutput.length > 0) {
                const splitOutput = doc.splitTextToSize(cleanOutput, 160);
                const outputHeight = Math.min((splitOutput.length * 4) + 6, 50);

                if (yPos + outputHeight + 5 > 275) {
                    doc.addPage();
                    yPos = 20;
                }

                // Light blue background for text output
                doc.setFillColor(245, 250, 255);
                doc.rect(25, yPos - 3, 160, outputHeight, 'F');

                doc.setFont("helvetica", "normal");
                doc.setFontSize(8);
                doc.setTextColor(0, 0, 0);
                doc.text(splitOutput, 27, yPos + 2);
                yPos += outputHeight + 8;
            }
        }

        yPos += 5;
    }

    doc.save("Logistic_Regression_Experiment.pdf");
}


// Init
function init() {
    renderSidebar();
    loadStep(0);

    // Attach Download Listener
    const downloadBtn = document.querySelector('.download-btn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadPDF);
    }
}

init();
